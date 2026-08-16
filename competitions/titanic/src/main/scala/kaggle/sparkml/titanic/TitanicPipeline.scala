package kaggle.sparkml.titanic

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, SQLTransformer, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}

object TitanicPipeline {
  private val BaselineExperiment = "baseline_001"
  private val DefaultExperiment = "rf_tuned_v2"
  private val AllExperiments = Seq(BaselineExperiment, DefaultExperiment, "gbt_v1", "lr_v1", "xgb_v1", "ensemble_v1")

  final case class Config(
      trainPath: String = "data/raw/train.csv",
      testPath: String = "data/raw/test.csv",
      outputPath: String = "output/submission.csv",
      modelPath: String = "models",
      experiment: String = DefaultExperiment,
      cvFolds: Int = 5,
      seed: Long = 42L,
      fast: Boolean = false,
      listExperiments: Boolean = false
  )

  final case class Candidate(name: String, pipeline: Pipeline, params: ParamMap = ParamMap.empty)
  final case class Metrics(accuracy: Double, auc: Double)
  final case class CandidateScore(name: String, accuracy: Double, auc: Double)
  final case class RunResult(
      experiment: String,
      candidate: String,
      accuracy: Double,
      auc: Double,
      modelPath: String,
      submissionPath: String
  )

  def main(args: Array[String]): Unit = {
    val config = parseArgs(args.toList, Config())

    if (config.listExperiments) {
      AllExperiments.foreach(println)
      return
    }

    require(config.cvFolds >= 2, "--cv-folds must be at least 2")
    require(config.experiment == "all" || AllExperiments.contains(config.experiment), s"Unknown experiment: ${config.experiment}")

    val spark = SparkSession
      .builder()
      .appName("kaggle-titanic-spark")
      .master(sys.props.getOrElse("spark.master", "local[*]"))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      val rawTrain = readCsv(spark, config.trainPath)
      val rawTest = readCsv(spark, config.testPath)
      val (enhancedTrain, enhancedTest) = buildEnhancedFrames(rawTrain, rawTest)

      val results =
        if (config.experiment == "all") {
          val completed = AllExperiments.flatMap { experiment =>
            try {
              Some(runExperiment(experiment, rawTrain, rawTest, enhancedTrain, enhancedTest, config, submissionFor(experiment), modelFor(experiment)))
            } catch {
              case e: Throwable if experiment == "xgb_v1" || experiment == "ensemble_v1" =>
                System.err.println(s"Skipping $experiment because XGBoost failed: ${e.getMessage}")
                None
            }
          }

          writeResultsCsv(spark, completed, "output/experiments.csv")
          completed.sortBy(r => (-r.accuracy, -r.auc)).headOption.foreach { best =>
            val bestSubmission = spark.read.option("header", "true").csv(best.submissionPath)
            writeSingleCsv(bestSubmission, config.outputPath)
            println(s"Best experiment by CV accuracy/AUC: ${best.experiment} (${best.candidate})")
          }
          completed
        } else {
          Seq(runExperiment(config.experiment, rawTrain, rawTest, enhancedTrain, enhancedTest, config, config.outputPath, modelFor(config.experiment, config.modelPath)))
        }

      results.foreach { r =>
        println(f"${r.experiment}%-13s ${r.candidate}%-48s accuracy=${r.accuracy}%.4f auc=${r.auc}%.4f submission=${r.submissionPath}")
      }
    } finally {
      spark.stop()
    }
  }

  private def runExperiment(
      experiment: String,
      rawTrain: DataFrame,
      rawTest: DataFrame,
      enhancedTrain: DataFrame,
      enhancedTest: DataFrame,
      config: Config,
      submissionPath: String,
      modelPath: String
  ): RunResult =
    experiment match {
      case BaselineExperiment =>
        runModelExperiment(experiment, rawTrain, rawTest, baselineCandidates(config.seed), config.cvFolds, modelPath, submissionPath)
      case "rf_tuned_v2" =>
        runModelExperiment(experiment, enhancedTrain, enhancedTest, maybeFast(rfCandidates(config.seed), config), config.cvFolds, modelPath, submissionPath)
      case "gbt_v1" =>
        runModelExperiment(experiment, enhancedTrain, enhancedTest, maybeFast(gbtCandidates(config.seed), config), config.cvFolds, modelPath, submissionPath)
      case "lr_v1" =>
        runModelExperiment(experiment, enhancedTrain, enhancedTest, maybeFast(lrCandidates(), config), config.cvFolds, modelPath, submissionPath)
      case "xgb_v1" =>
        runModelExperiment(experiment, enhancedTrain, enhancedTest, maybeFast(xgbCandidates(config.seed), config), config.cvFolds, modelPath, submissionPath)
      case "ensemble_v1" =>
        runEnsembleExperiment(enhancedTrain, enhancedTest, config, modelPath, submissionPath)
    }

  private def runModelExperiment(
      experiment: String,
      train: DataFrame,
      test: DataFrame,
      candidates: Seq[Candidate],
      folds: Int,
      modelPath: String,
      submissionPath: String
  ): RunResult = {
    val scores = candidates.map(c => scoreCandidate(train, c, folds))
    scores.foreach(s => println(f"$experiment ${s.name} accuracy=${s.accuracy}%.4f auc=${s.auc}%.4f"))
    val bestScore = scores.sortBy(s => (-s.accuracy, -s.auc)).head
    val best = candidates.find(_.name == bestScore.name).get
    val model = best.pipeline.fit(train, best.params)

    model.write.overwrite().save(modelPath)
    writeSubmission(model.transform(test), submissionPath)

    RunResult(experiment, best.name, bestScore.accuracy, bestScore.auc, modelPath, submissionPath)
  }

  private def runEnsembleExperiment(
      train: DataFrame,
      test: DataFrame,
      config: Config,
      modelPath: String,
      submissionPath: String
  ): RunResult = {
    val members = Seq(
      maybeFast(rfCandidates(config.seed), config).head,
      maybeFast(gbtCandidates(config.seed), config).head,
      maybeFast(xgbCandidates(config.seed), config).head
    )

    val folded = withFolds(train, config.cvFolds)
    val foldMetrics = (0 until config.cvFolds).map { fold =>
      val training = folded.filter(col("__fold") =!= lit(fold)).drop("__fold")
      val validation = folded.filter(col("__fold") === lit(fold)).drop("__fold")
      metrics(ensemblePredict(members, training, validation))
    }
    val accuracy = foldMetrics.map(_.accuracy).sum / foldMetrics.size
    val auc = foldMetrics.map(_.auc).sum / foldMetrics.size

    writeSubmission(ensemblePredict(members, train, test), submissionPath)
    members.zip(Seq("rf_tuned_v2", "gbt_v1", "xgb_v1")).foreach { case (candidate, name) =>
      candidate.pipeline.fit(train, candidate.params).write.overwrite().save(s"${modelPath}_$name")
    }

    RunResult("ensemble_v1", "soft_vote_rf_gbt_xgb", accuracy, auc, modelPath, submissionPath)
  }

  private def ensemblePredict(candidates: Seq[Candidate], train: DataFrame, data: DataFrame): DataFrame = {
    val probabilityAtOne = udf((v: Vector) => v(1))
    val rawPredictionVector = udf((p: Double) => Vectors.dense(1.0 - p, p))
    val probabilityFrames = candidates.zipWithIndex.map { case (candidate, index) =>
      val model = candidate.pipeline.fit(train, candidate.params)
      model
        .transform(data)
        .select(col("PassengerId"), probabilityAtOne(col("probability")).as(s"p$index"))
    }

    val joined = probabilityFrames.reduce(_.join(_, Seq("PassengerId"), "inner"))
    val avgProbability = candidates.indices.map(i => col(s"p$i")).reduce(_ + _) / lit(candidates.size.toDouble)
    val keepColumns = if (data.columns.contains("Survived")) Seq("PassengerId", "Survived") else Seq("PassengerId")

    data
      .select(keepColumns.head, keepColumns.tail: _*)
      .join(joined, Seq("PassengerId"), "inner")
      .withColumn("probability_one", avgProbability)
      .withColumn("rawPrediction", rawPredictionVector(col("probability_one")))
      .withColumn("prediction", when(col("probability_one") >= 0.5, 1.0).otherwise(0.0))
  }

  private def scoreCandidate(train: DataFrame, candidate: Candidate, folds: Int): CandidateScore = {
    val folded = withFolds(train, folds)
    val foldMetrics = (0 until folds).map { fold =>
      val training = folded.filter(col("__fold") =!= lit(fold)).drop("__fold")
      val validation = folded.filter(col("__fold") === lit(fold)).drop("__fold")
      val model = candidate.pipeline.fit(training, candidate.params)
      metrics(model.transform(validation))
    }

    CandidateScore(
      candidate.name,
      foldMetrics.map(_.accuracy).sum / foldMetrics.size,
      foldMetrics.map(_.auc).sum / foldMetrics.size
    )
  }

  private def maybeFast(candidates: Seq[Candidate], config: Config): Seq[Candidate] =
    if (config.fast) candidates.take(1) else candidates

  private def metrics(predictions: DataFrame): Metrics = {
    val accuracy = predictions
      .select((col("prediction") === col("Survived").cast("double")).cast("double").as("ok"))
      .agg(avg("ok").as("accuracy"))
      .first()
      .getAs[Double]("accuracy")

    val auc = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
      .evaluate(predictions)

    Metrics(accuracy, auc)
  }

  private def withFolds(df: DataFrame, folds: Int): DataFrame =
    df.withColumn("__fold", pmod(abs(hash(col("PassengerId"))), lit(folds)).cast(IntegerType))

  private def readCsv(spark: SparkSession, path: String): DataFrame =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)

  private def buildEnhancedFrames(rawTrain: DataFrame, rawTest: DataFrame): (DataFrame, DataFrame) = {
    val train = rawTrain.withColumn("__isTrain", lit(true))
    val test = rawTest
      .withColumn("Survived", lit(null).cast(IntegerType))
      .withColumn("__isTrain", lit(false))

    val combined = train.unionByName(test, allowMissingColumns = true)
    val base = combined
      .withColumn("Pclass", col("Pclass").cast(DoubleType))
      .withColumn("SibSp", col("SibSp").cast(DoubleType))
      .withColumn("Parch", col("Parch").cast(DoubleType))
      .withColumn("Age", col("Age").cast(DoubleType))
      .withColumn("Fare", col("Fare").cast(DoubleType))
      .withColumn("RawTitle", regexp_extract(col("Name"), ",\\s*([^\\.]+)\\.", 1))
      .withColumn(
        "TitleGroup",
        when(col("RawTitle").isin("Mr", "Mrs", "Miss", "Master"), col("RawTitle"))
          .when(col("RawTitle").isin("Mlle", "Ms"), lit("Miss"))
          .when(col("RawTitle") === "Mme", lit("Mrs"))
          .otherwise(lit("Rare"))
      )
      .withColumn("Surname", regexp_extract(col("Name"), "^([^,]+),", 1))
      .withColumn("EmbarkedFilledBase", coalesce(col("Embarked"), lit("S")))
      .withColumn("Deck", when(col("Cabin").isNull || length(trim(col("Cabin"))) === 0, lit("missing")).otherwise(substring(col("Cabin"), 1, 1)))
      .withColumn("HasCabin", when(col("Deck") === "missing", 0.0).otherwise(1.0))
      .withColumn("CabinCount", when(col("Deck") === "missing", 0.0).otherwise(size(split(trim(col("Cabin")), "\\s+")).cast(DoubleType)))
      .withColumn("TicketPrefixRaw", regexp_replace(regexp_extract(upper(col("Ticket")), "^([^0-9]+)", 1), "[^A-Z]", ""))
      .withColumn("TicketPrefix", when(length(col("TicketPrefixRaw")) === 0, lit("NUMERIC")).otherwise(col("TicketPrefixRaw")))
      .withColumn("IsNumericTicket", when(col("TicketPrefix") === "NUMERIC", 1.0).otherwise(0.0))
      .withColumn("FamilySize", col("SibSp") + col("Parch") + lit(1.0))
      .withColumn(
        "FamilyBucket",
        when(col("FamilySize") === 1.0, lit("single"))
          .when(col("FamilySize") <= 3.0, lit("small"))
          .when(col("FamilySize") <= 5.0, lit("medium"))
          .otherwise(lit("large"))
      )
      .withColumn("IsAlone", when(col("FamilySize") === 1.0, 1.0).otherwise(0.0))
      .withColumn("AgeKnown", when(col("Age").isNull, 0.0).otherwise(1.0))
      .withColumn("FareKnown", when(col("Fare").isNull, 0.0).otherwise(1.0))

    val surnameCounts = base.groupBy("Surname").agg(count(lit(1)).cast(DoubleType).as("SurnameGroupSize"))
    val ticketCounts = base.groupBy("Ticket").agg(count(lit(1)).cast(DoubleType).as("TicketGroupSize"))
    val globalAge = medianOrDefault(base, "Age", 28.0)
    val globalFare = medianOrDefault(base, "Fare", 14.4542)

    val ageByTitle = base
      .filter(col("Age").isNotNull)
      .groupBy("Sex", "Pclass", "TitleGroup")
      .agg(expr("percentile_approx(Age, 0.5)").cast(DoubleType).as("AgeMedianTitle"))

    val ageByClass = base
      .filter(col("Age").isNotNull)
      .groupBy("Sex", "Pclass")
      .agg(expr("percentile_approx(Age, 0.5)").cast(DoubleType).as("AgeMedianClass"))

    val fareByEmbarked = base
      .filter(col("Fare").isNotNull)
      .groupBy("Pclass", "EmbarkedFilledBase")
      .agg(expr("percentile_approx(Fare, 0.5)").cast(DoubleType).as("FareMedianEmbarked"))

    val fareByClass = base
      .filter(col("Fare").isNotNull)
      .groupBy("Pclass")
      .agg(expr("percentile_approx(Fare, 0.5)").cast(DoubleType).as("FareMedianClass"))

    val enriched = base
      .join(surnameCounts, Seq("Surname"), "left")
      .join(ticketCounts, Seq("Ticket"), "left")
      .join(ageByTitle, Seq("Sex", "Pclass", "TitleGroup"), "left")
      .join(ageByClass, Seq("Sex", "Pclass"), "left")
      .join(fareByEmbarked, Seq("Pclass", "EmbarkedFilledBase"), "left")
      .join(fareByClass, Seq("Pclass"), "left")
      .withColumn("AgeFilled", coalesce(col("Age"), col("AgeMedianTitle"), col("AgeMedianClass"), lit(globalAge)).cast(DoubleType))
      .withColumn("FareFilled", coalesce(col("Fare"), col("FareMedianEmbarked"), col("FareMedianClass"), lit(globalFare)).cast(DoubleType))
      .withColumn("EmbarkedFilled", col("EmbarkedFilledBase"))
      .withColumn("FarePerPerson", col("FareFilled") / when(col("FamilySize") <= 0.0, lit(1.0)).otherwise(col("FamilySize")))
      .withColumn("FareLog1p", log1p(col("FareFilled")))
      .withColumn("PclassText", col("Pclass").cast(IntegerType).cast("string"))
      .drop("RawTitle", "TicketPrefixRaw", "AgeMedianTitle", "AgeMedianClass", "FareMedianEmbarked", "FareMedianClass")

    val trainOut = enriched.filter(col("__isTrain")).drop("__isTrain")
    val testOut = enriched.filter(!col("__isTrain")).drop("__isTrain", "Survived")
    (trainOut, testOut)
  }

  private def medianOrDefault(df: DataFrame, column: String, default: Double): Double = {
    val values = df.stat.approxQuantile(column, Array(0.5), 0.001)
    values.headOption.getOrElse(default)
  }

  private def baselineCandidates(seed: Long): Seq[Candidate] =
    Seq(Candidate("baseline_rf_200_depth_6", baselinePipeline(seed)))

  private def baselinePipeline(seed: Long): Pipeline = {
    val baseFeatures = sqlTransformer(
      """
        |SELECT
        |  *,
        |  CAST(Pclass AS STRING) AS PclassText,
        |  COALESCE(Embarked, 'missing') AS EmbarkedText,
        |  CAST(SibSp + Parch + 1 AS DOUBLE) AS FamilySize,
        |  CASE WHEN SibSp + Parch = 0 THEN 1.0 ELSE 0.0 END AS IsAlone,
        |  COALESCE(regexp_extract(Name, ',\\s*([^\\.]+)\\.', 1), 'missing') AS Title
        |FROM __THIS__
        |""".stripMargin
    )

    val imputer = new Imputer()
      .setInputCols(Array("Age", "Fare"))
      .setOutputCols(Array("AgeImputed", "FareImputed"))
      .setStrategy("median")

    val derivedFeatures = sqlTransformer(
      """
        |SELECT
        |  *,
        |  FareImputed / CASE WHEN FamilySize <= 0.0 THEN 1.0 ELSE FamilySize END AS FarePerPerson
        |FROM __THIS__
        |""".stripMargin
    )

    val indexers = Array(
      indexer("Sex", "SexIndexed"),
      indexer("EmbarkedText", "EmbarkedIndexed"),
      indexer("PclassText", "PclassIndexed"),
      indexer("Title", "TitleIndexed")
    )

    val assembler = new VectorAssembler()
      .setInputCols(Array("PclassIndexed", "SexIndexed", "EmbarkedIndexed", "TitleIndexed", "AgeImputed", "FareImputed", "SibSp", "Parch", "FamilySize", "IsAlone", "FarePerPerson"))
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val classifier = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setSeed(seed)
      .setNumTrees(200)
      .setMaxDepth(6)

    new Pipeline().setStages(Array[PipelineStage](baseFeatures, imputer, derivedFeatures) ++ indexers ++ Array(assembler, classifier))
  }

  private def rfCandidates(seed: Long): Seq[Candidate] =
    for {
      numTrees <- Seq(300, 600)
      maxDepth <- Seq(5, 7, 9)
      minInstances <- Seq(1, 2)
      subsamplingRate <- Seq(0.8, 1.0)
    } yield {
      val rf = new RandomForestClassifier()
        .setLabelCol("Survived")
        .setFeaturesCol("features")
        .setSeed(seed)
        .setFeatureSubsetStrategy("sqrt")
      Candidate(
        s"rf_trees_${numTrees}_depth_${maxDepth}_min_${minInstances}_sub_${subsamplingRate}",
        treePipeline(rf),
        new ParamMap()
          .put(rf.numTrees, numTrees)
          .put(rf.maxDepth, maxDepth)
          .put(rf.minInstancesPerNode, minInstances)
          .put(rf.subsamplingRate, subsamplingRate)
      )
    }

  private def gbtCandidates(seed: Long): Seq[Candidate] =
    for {
      maxIter <- Seq(80, 150)
      maxDepth <- Seq(2, 3, 4)
      stepSize <- Seq(0.03, 0.05, 0.1)
    } yield {
      val gbt = new GBTClassifier()
        .setLabelCol("Survived")
        .setFeaturesCol("features")
        .setSeed(seed)
      Candidate(
        s"gbt_iter_${maxIter}_depth_${maxDepth}_step_${stepSize}",
        treePipeline(gbt),
        new ParamMap()
          .put(gbt.maxIter, maxIter)
          .put(gbt.maxDepth, maxDepth)
          .put(gbt.stepSize, stepSize)
      )
    }

  private def lrCandidates(): Seq[Candidate] =
    for {
      regParam <- Seq(0.01, 0.05, 0.1)
      elasticNet <- Seq(0.0, 0.5, 1.0)
    } yield {
      val lr = new LogisticRegression()
        .setLabelCol("Survived")
        .setFeaturesCol("features")
        .setMaxIter(200)
        .setStandardization(false)
      Candidate(
        s"lr_reg_${regParam}_elastic_${elasticNet}",
        linearPipeline(lr),
        new ParamMap()
          .put(lr.regParam, regParam)
          .put(lr.elasticNetParam, elasticNet)
      )
    }

  private def xgbCandidates(seed: Long): Seq[Candidate] =
    for {
      maxDepth <- Seq(2, 3, 4)
      eta <- Seq(0.03, 0.05)
      numRound <- Seq(100, 200)
    } yield {
      val xgb = xgbClassifier(seed, maxDepth, eta, numRound, xgbFeatureCols)
      Candidate(s"xgb_round_${numRound}_depth_${maxDepth}_eta_${eta}", xgbPipeline(xgb))
    }

  private def treePipeline(classifier: PipelineStage): Pipeline = {
    val categorical = Array("Sex", "EmbarkedFilled", "PclassText", "TitleGroup", "Deck", "TicketPrefix", "FamilyBucket")
    val indexers = categorical.map(name => indexer(name, s"${name}Indexed"))
    val assembler = new VectorAssembler()
      .setInputCols(categorical.map(name => s"${name}Indexed") ++ numericFeatures)
      .setOutputCol("features")
      .setHandleInvalid("keep")

    new Pipeline().setStages(indexers ++ Array[PipelineStage](assembler, classifier))
  }

  private def linearPipeline(classifier: PipelineStage): Pipeline = {
    val categorical = Array("Sex", "EmbarkedFilled", "PclassText", "TitleGroup", "Deck", "TicketPrefix", "FamilyBucket")
    val indexers = categorical.map(name => indexer(name, s"${name}Indexed"))
    val encoder = new OneHotEncoder()
      .setInputCols(categorical.map(name => s"${name}Indexed"))
      .setOutputCols(categorical.map(name => s"${name}OneHot"))
      .setHandleInvalid("keep")
    val numericAssembler = new VectorAssembler()
      .setInputCols(numericFeatures)
      .setOutputCol("numericFeatures")
      .setHandleInvalid("keep")
    val scaler = new StandardScaler()
      .setInputCol("numericFeatures")
      .setOutputCol("scaledNumericFeatures")
      .setWithMean(true)
      .setWithStd(true)
    val assembler = new VectorAssembler()
      .setInputCols(categorical.map(name => s"${name}OneHot") :+ "scaledNumericFeatures")
      .setOutputCol("features")
      .setHandleInvalid("keep")

    new Pipeline().setStages(indexers ++ Array[PipelineStage](encoder, numericAssembler, scaler, assembler, classifier))
  }

  private def xgbPipeline(classifier: PipelineStage): Pipeline = {
    val categorical = Array("Sex", "EmbarkedFilled", "PclassText", "TitleGroup", "Deck", "TicketPrefix", "FamilyBucket")
    val indexers = categorical.map(name => indexer(name, s"${name}Indexed"))
    new Pipeline().setStages(indexers ++ Array(classifier))
  }

  private def numericFeatures: Array[String] =
    Array(
      "AgeFilled",
      "FareFilled",
      "FarePerPerson",
      "FareLog1p",
      "SibSp",
      "Parch",
      "FamilySize",
      "IsAlone",
      "SurnameGroupSize",
      "TicketGroupSize",
      "HasCabin",
      "CabinCount",
      "IsNumericTicket",
      "AgeKnown",
      "FareKnown"
    )

  private def xgbFeatureCols: Array[String] =
    Array("SexIndexed", "EmbarkedFilledIndexed", "PclassTextIndexed", "TitleGroupIndexed", "DeckIndexed", "TicketPrefixIndexed", "FamilyBucketIndexed") ++ numericFeatures

  private def xgbClassifier(seed: Long, maxDepth: Int, eta: Double, numRound: Int, featuresCols: Array[String]): PipelineStage = {
    val cls = Class.forName("ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier")
    val xgb = cls.getConstructor().newInstance().asInstanceOf[AnyRef]
    callSetter(xgb, "setFeaturesCol", featuresCols)
    callSetter(xgb, "setLabelCol", "Survived")
    callSetter(xgb, "setPredictionCol", "prediction")
    callSetter(xgb, "setProbabilityCol", "probability")
    callSetter(xgb, "setRawPredictionCol", "rawPrediction")
    callSetter(xgb, "setObjective", "binary:logistic")
    callSetter(xgb, "setNumWorkers", Int.box(1))
    callSetter(xgb, "setMaxDepth", Int.box(maxDepth))
    callSetter(xgb, "setEta", Double.box(eta))
    callSetter(xgb, "setNumRound", Int.box(numRound))
    callSetter(xgb, "setSeed", Long.box(seed))
    callSetter(xgb, "setVerbosity", Int.box(0))
    xgb.asInstanceOf[PipelineStage]
  }

  private def callSetter(instance: AnyRef, name: String, value: AnyRef): Unit = {
    val method = instance.getClass.getMethods.find(m => m.getName == name && m.getParameterCount == 1).getOrElse {
      throw new NoSuchMethodException(s"${instance.getClass.getName}.$name")
    }
    method.invoke(instance, value)
  }

  private def indexer(inputCol: String, outputCol: String): StringIndexer =
    new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setHandleInvalid("keep")

  private def sqlTransformer(statement: String): SQLTransformer =
    new SQLTransformer().setStatement(statement)

  private def writeSubmission(predictions: DataFrame, outputPath: String): Unit = {
    val submission = predictions
      .select(
        col("PassengerId").cast(IntegerType).as("PassengerId"),
        col("prediction").cast(IntegerType).as("Survived")
      )
      .orderBy("PassengerId")

    writeSingleCsv(submission, outputPath)
  }

  private def writeResultsCsv(spark: SparkSession, results: Seq[RunResult], outputPath: String): Unit = {
    import spark.implicits._
    writeSingleCsv(results.toDF(), outputPath)
  }

  private def writeSingleCsv(df: DataFrame, outputPath: String): Unit = {
    val spark = df.sparkSession
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val fs = FileSystem.get(hadoopConf)
    val output = new Path(outputPath)
    val temp = new Path(s"$outputPath.spark-tmp-${System.currentTimeMillis()}")

    df.coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(temp.toString)

    val partFile = fs.globStatus(new Path(temp, "part-*.csv")).headOption.map(_.getPath).getOrElse {
      throw new IllegalStateException(s"No Spark part file found under $temp")
    }

    val parent = output.getParent
    if (parent != null && !fs.exists(parent)) {
      fs.mkdirs(parent)
    }
    if (fs.exists(output)) {
      fs.delete(output, false)
    }
    fs.rename(partFile, output)
    fs.delete(temp, true)
  }

  private def submissionFor(experiment: String): String =
    s"output/submission_$experiment.csv"

  private def modelFor(experiment: String, root: String = "models"): String =
    s"$root/$experiment"

  private def parseArgs(args: List[String], config: Config): Config =
    args match {
      case Nil => config
      case "--train" :: value :: tail => parseArgs(tail, config.copy(trainPath = value))
      case "--test" :: value :: tail => parseArgs(tail, config.copy(testPath = value))
      case "--output" :: value :: tail => parseArgs(tail, config.copy(outputPath = value))
      case "--model" :: value :: tail => parseArgs(tail, config.copy(modelPath = value))
      case "--experiment" :: value :: tail => parseArgs(tail, config.copy(experiment = value))
      case "--cv-folds" :: value :: tail => parseArgs(tail, config.copy(cvFolds = value.toInt))
      case "--seed" :: value :: tail => parseArgs(tail, config.copy(seed = value.toLong))
      case "--fast" :: tail => parseArgs(tail, config.copy(fast = true))
      case "--list-experiments" :: tail => parseArgs(tail, config.copy(listExperiments = true))
      case "--help" :: _ =>
        println(
          s"""
             |Usage:
             |  titanic/run [--experiment ${AllExperiments.mkString("|")}|all] [--cv-folds n] [--fast] [--train path] [--test path] [--output path] [--model path] [--seed n]
             |
             |Default experiment: $DefaultExperiment
             |""".stripMargin.trim
        )
        sys.exit(0)
      case flag :: _ if flag.startsWith("--") =>
        throw new IllegalArgumentException(s"Unknown argument: $flag")
      case value :: _ =>
        throw new IllegalArgumentException(s"Unexpected positional argument: $value")
    }
}
