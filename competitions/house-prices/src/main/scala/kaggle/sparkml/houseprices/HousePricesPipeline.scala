package kaggle.sparkml.houseprices

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.{Pipeline, UnaryTransformer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, NumericType, StringType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

final class DenseVectorTransformer(override val uid: String)
    extends UnaryTransformer[Vector, Vector, DenseVectorTransformer]
    with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("denseVector"))

  override protected def createTransformFunc: Vector => Vector = _.toDense
  override protected def outputDataType: DataType = SQLDataTypes.VectorType
}

object DenseVectorTransformer extends DefaultParamsReadable[DenseVectorTransformer]

object HousePricesPipeline {
  private val TargetCol = "SalePrice"
  private val LabelCol = "SalePriceLog"
  private val IdCol = "Id"
  private val NumericAsCategorical = Set("MSSubClass", "MoSold", "YrSold")

  final case class Config(
      trainPath: String = "data/raw/train.csv",
      testPath: String = "data/raw/test.csv",
      outputPath: String = "output/submission.csv",
      modelPath: String = "models/lr_elastic_net_v1",
      algo: String = "lr",
      maxBins: Int = 512,
      numTrees: Int = 300,
      maxDepth: Int = 6,
      maxIter: Int = 250,
      regParam: Double = 0.01,
      elasticNetParam: Double = 0.8,
      xgbRounds: Int = 500,
      xgbEta: Double = 0.03,
      ensembleXgbWeight: Double = 0.4,
      advancedFeatures: Boolean = false,
      removeOutliers: Boolean = false,
      validationOnly: Boolean = false,
      validationSeeds: Seq[Long] = Seq.empty,
      seed: Long = 42L
  )

  def main(args: Array[String]): Unit = {
    val config = parseArgs(args.toList, Config())
    require(Set("lr", "rf", "gbt", "xgb", "ensemble").contains(config.algo), s"Unknown --algo: ${config.algo}")
    require(config.maxBins >= 32, "--max-bins must be at least 32")
    require(config.xgbRounds > 0, "--xgb-rounds must be positive")
    require(config.xgbEta > 0.0, "--xgb-eta must be positive")
    require(config.ensembleXgbWeight >= 0.0 && config.ensembleXgbWeight <= 1.0, "--ensemble-xgb-weight must be between 0 and 1")

    val spark = SparkSession
      .builder()
      .appName("kaggle-house-prices-spark")
      .master(sys.props.getOrElse("spark.master", "local[*]"))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      val rawTrain = withHouseFeatures(readCsv(spark, config.trainPath), config.advancedFeatures)
      val rawTest = withHouseFeatures(readCsv(spark, config.testPath), config.advancedFeatures)
      val train = rawTrain.withColumn(LabelCol, log1p(col(TargetCol).cast(DoubleType)))

      val featureColumns = train.columns.filterNot(Set(IdCol, TargetCol, LabelCol))
      val numericColumns = train.schema.fields.collect {
        case field if featureColumns.contains(field.name) && isNumeric(field.dataType) && !NumericAsCategorical.contains(field.name) => field.name
      }
      val categoricalColumns = featureColumns.diff(numericColumns)

      println(s"Algorithm: ${config.algo}")
      println(s"Max bins: ${config.maxBins}")
      println(s"Numeric features: ${numericColumns.length}")
      println(s"Categorical features: ${categoricalColumns.length}")
      println(s"Advanced features: ${config.advancedFeatures}")
      println(s"Remove anomalous large-house rows from training: ${config.removeOutliers}")
      println(s"Validation only: ${config.validationOnly}")

      val preparedTrain = withCategoricalText(train, categoricalColumns)
      val preparedTest = withCategoricalText(rawTest, categoricalColumns)
      val seeds = if (config.validationSeeds.nonEmpty) config.validationSeeds else Seq(config.seed)
      if (config.algo == "ensemble") {
        runEnsemble(numericColumns, categoricalColumns, preparedTrain, preparedTest, seeds, config)
      } else {
        runSingleModel(numericColumns, categoricalColumns, preparedTrain, preparedTest, seeds, config)
      }
    } finally {
      spark.stop()
    }
  }

  private def runSingleModel(
      numericColumns: Array[String],
      categoricalColumns: Array[String],
      preparedTrain: DataFrame,
      preparedTest: DataFrame,
      seeds: Seq[Long],
      config: Config
  ): Unit = {
    val pipeline = buildPipeline(numericColumns, categoricalColumns, config)
    val validationScores = seeds.map { seed =>
      val Array(trainingWithOutliers, validation) = preparedTrain.randomSplit(Array(0.8, 0.2), seed)
      val training = filterTrainingOutliers(trainingWithOutliers, config.removeOutliers)
      val validationModel = pipeline.fit(training)
      val validationPredictions = validationModel.transform(validation)
      val logRmse = evaluateLogRmse(validationPredictions)
      println(f"Validation log-RMSE (seed $seed): $logRmse%.5f")
      logRmse
    }

    printValidationSummary(validationScores)

    if (!config.validationOnly) {
      val finalModel = pipeline.fit(filterTrainingOutliers(preparedTrain, config.removeOutliers))
      finalModel.write.overwrite().save(config.modelPath)
      writeSubmission(finalModel.transform(preparedTest), config.outputPath)
      println(s"Model saved: ${config.modelPath}")
      println(s"Submission saved: ${config.outputPath}")
    }
  }

  private def runEnsemble(
      numericColumns: Array[String],
      categoricalColumns: Array[String],
      preparedTrain: DataFrame,
      preparedTest: DataFrame,
      seeds: Seq[Long],
      config: Config
  ): Unit = {
    val weights = Seq(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    val scoresBySeed = seeds.map { seed =>
      val Array(trainingWithOutliers, validation) = preparedTrain.randomSplit(Array(0.8, 0.2), seed)
      val training = filterTrainingOutliers(trainingWithOutliers, config.removeOutliers)
      val lrModel = buildPipeline(numericColumns, categoricalColumns, config.copy(algo = "lr", seed = seed)).fit(training)
      val xgbModel = buildPipeline(numericColumns, categoricalColumns, config.copy(algo = "xgb", seed = seed)).fit(training)
      val joined = joinPredictions(lrModel.transform(validation), xgbModel.transform(validation), includeLabel = true)
      val scoreColumns = weights.map { weight =>
        val prediction = col("lr_prediction") * lit(1.0 - weight) + col("xgb_prediction") * lit(weight)
        sqrt(avg(pow(col(LabelCol) - prediction, 2.0))).as(weightColumn(weight))
      }
      val row = joined.agg(scoreColumns.head, scoreColumns.tail: _*).head()
      val scores = weights.zipWithIndex.map { case (weight, index) => weight -> row.getDouble(index) }.toMap
      println(s"Ensemble validation seed: $seed")
      weights.foreach(weight => println(f"  XGBoost weight $weight%.1f: ${scores(weight)}%.5f"))
      scores
    }

    if (scoresBySeed.lengthCompare(1) > 0) {
      println("Ensemble mean validation log-RMSE:")
      weights.foreach { weight =>
        val scores = scoresBySeed.map(_(weight))
        val mean = scores.sum / scores.size
        val stddev = math.sqrt(scores.map(score => math.pow(score - mean, 2.0)).sum / scores.size)
        println(f"  XGBoost weight $weight%.1f: $mean%.5f (stddev $stddev%.5f)")
      }
    }

    if (!config.validationOnly) {
      val finalTrain = filterTrainingOutliers(preparedTrain, config.removeOutliers)
      val lrModel = buildPipeline(numericColumns, categoricalColumns, config.copy(algo = "lr")).fit(finalTrain)
      val xgbModel = buildPipeline(numericColumns, categoricalColumns, config.copy(algo = "xgb")).fit(finalTrain)
      lrModel.write.overwrite().save(s"${config.modelPath}_lr")
      xgbModel.write.overwrite().save(s"${config.modelPath}_xgb")
      val joined = joinPredictions(lrModel.transform(preparedTest), xgbModel.transform(preparedTest), includeLabel = false)
        .withColumn(
          "prediction",
          col("lr_prediction") * lit(1.0 - config.ensembleXgbWeight) +
            col("xgb_prediction") * lit(config.ensembleXgbWeight)
        )
      writeSubmission(joined, config.outputPath)
      println(f"Ensemble XGBoost weight: ${config.ensembleXgbWeight}%.2f")
      println(s"Models saved: ${config.modelPath}_lr and ${config.modelPath}_xgb")
      println(s"Submission saved: ${config.outputPath}")
    }
  }

  private def joinPredictions(lrPredictions: DataFrame, xgbPredictions: DataFrame, includeLabel: Boolean): DataFrame = {
    val lrColumns =
      if (includeLabel) Seq(col(IdCol), col(LabelCol), col("prediction").as("lr_prediction"))
      else Seq(col(IdCol), col("prediction").as("lr_prediction"))
    lrPredictions
      .select(lrColumns: _*)
      .join(xgbPredictions.select(col(IdCol), col("prediction").as("xgb_prediction")), Seq(IdCol), "inner")
  }

  private def evaluateLogRmse(predictions: DataFrame): Double =
    new RegressionEvaluator()
      .setLabelCol(LabelCol)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      .evaluate(predictions)

  private def printValidationSummary(scores: Seq[Double]): Unit =
    if (scores.lengthCompare(1) > 0) {
      val mean = scores.sum / scores.size
      val stddev = math.sqrt(scores.map(score => math.pow(score - mean, 2.0)).sum / scores.size)
      println(f"Validation log-RMSE mean: $mean%.5f")
      println(f"Validation log-RMSE stddev: $stddev%.5f")
    }

  private def weightColumn(weight: Double): String = f"rmse_xgb_${weight * 100.0}%.0f"

  private def buildPipeline(numericColumns: Array[String], categoricalColumns: Array[String], config: Config): Pipeline = {
    val imputedNumericColumns = numericColumns.map(c => s"${c}_imputed")
    val indexedCategoricalColumns = categoricalColumns.map(c => s"${c}_indexed")

    val imputerStages =
      if (numericColumns.nonEmpty) {
        Seq(
          new Imputer()
            .setStrategy("median")
            .setInputCols(numericColumns)
            .setOutputCols(imputedNumericColumns)
        )
      } else {
        Seq.empty[PipelineStage]
      }

    val indexerStages = categoricalColumns.map { c =>
      new StringIndexer()
        .setInputCol(textCol(c))
        .setOutputCol(s"${c}_indexed")
        .setHandleInvalid("keep")
    }

    val stages = config.algo match {
      case "lr" =>
        val encodedCategoricalColumns = categoricalColumns.map(c => s"${c}_onehot")
        val encoderStages =
          if (categoricalColumns.nonEmpty) {
            Seq(
              new OneHotEncoder()
                .setInputCols(indexedCategoricalColumns)
                .setOutputCols(encodedCategoricalColumns)
                .setDropLast(false)
                .setHandleInvalid("keep")
            )
          } else {
            Seq.empty[PipelineStage]
          }

        val numericAssembler = new VectorAssembler()
          .setInputCols(imputedNumericColumns)
          .setOutputCol("numeric_features")
          .setHandleInvalid("keep")

        val scaler = new StandardScaler()
          .setInputCol("numeric_features")
          .setOutputCol("numeric_scaled")
          .setWithMean(false)
          .setWithStd(true)

        val assembler = new VectorAssembler()
          .setInputCols(Array("numeric_scaled") ++ encodedCategoricalColumns)
          .setOutputCol("features")
          .setHandleInvalid("keep")

        val model = new LinearRegression()
          .setLabelCol(LabelCol)
          .setFeaturesCol("features")
          .setMaxIter(config.maxIter)
          .setRegParam(config.regParam)
          .setElasticNetParam(config.elasticNetParam)
          .setStandardization(true)

        imputerStages ++ indexerStages ++ encoderStages ++ Seq(numericAssembler, scaler, assembler, model)

      case "rf" =>
        val assembler = new VectorAssembler()
          .setInputCols(imputedNumericColumns ++ indexedCategoricalColumns)
          .setOutputCol("features")
          .setHandleInvalid("keep")

        val model = new RandomForestRegressor()
          .setLabelCol(LabelCol)
          .setFeaturesCol("features")
          .setNumTrees(config.numTrees)
          .setMaxDepth(config.maxDepth)
          .setMaxBins(config.maxBins)
          .setMinInstancesPerNode(2)
          .setSubsamplingRate(0.85)
          .setFeatureSubsetStrategy("sqrt")
          .setSeed(config.seed)

        imputerStages ++ indexerStages ++ Seq(assembler, model)

      case "gbt" =>
        val assembler = new VectorAssembler()
          .setInputCols(imputedNumericColumns ++ indexedCategoricalColumns)
          .setOutputCol("features")
          .setHandleInvalid("keep")

        val model = new GBTRegressor()
          .setLabelCol(LabelCol)
          .setFeaturesCol("features")
          .setMaxIter(config.maxIter)
          .setMaxDepth(config.maxDepth)
          .setMaxBins(config.maxBins)
          .setStepSize(0.05)
          .setSeed(config.seed)

        imputerStages ++ indexerStages ++ Seq(assembler, model)

      case "xgb" =>
        val assembler = new VectorAssembler()
          .setInputCols(imputedNumericColumns ++ indexedCategoricalColumns)
          .setOutputCol("features_sparse_or_dense")
          .setHandleInvalid("keep")

        val densifier = new DenseVectorTransformer()
          .setInputCol("features_sparse_or_dense")
          .setOutputCol("features")

        val model = new XGBoostRegressor()
          .setLabelCol(LabelCol)
          .setFeaturesCol("features")
          .setObjective("reg:squarederror")
          .setNumWorkers(1)
          .setNthread(1)
          .setNumRound(config.xgbRounds)
          .setEta(config.xgbEta)
          .setMaxDepth(config.maxDepth)
          .setMinChildWeight(1.0)
          .setSubsample(0.8)
          .setColsampleBytree(0.8)
          .setLambda(1.0)
          .setTreeMethod("hist")
          .setVerbosity(0)
          .setSeed(config.seed)

        imputerStages ++ indexerStages ++ Seq(assembler, densifier, model)
    }

    new Pipeline().setStages(stages.toArray)
  }

  private def readCsv(spark: SparkSession, path: String): DataFrame =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("nullValue", "NA")
      .csv(path)

  private def withHouseFeatures(df: DataFrame, advancedFeatures: Boolean): DataFrame = {
    val withAge = withColumnIfPresent(df, "HouseAgeAtSale", Seq("YrSold", "YearBuilt")) {
      col("YrSold").cast(DoubleType) - col("YearBuilt").cast(DoubleType)
    }
    val withRemodelAge = withColumnIfPresent(withAge, "RemodAgeAtSale", Seq("YrSold", "YearRemodAdd")) {
      col("YrSold").cast(DoubleType) - col("YearRemodAdd").cast(DoubleType)
    }
    val withTotalSf = withColumnIfPresent(withRemodelAge, "TotalSF", Seq("TotalBsmtSF", "1stFlrSF", "2ndFlrSF")) {
      coalesce(col("TotalBsmtSF").cast(DoubleType), lit(0.0)) +
        coalesce(col("1stFlrSF").cast(DoubleType), lit(0.0)) +
        coalesce(col("2ndFlrSF").cast(DoubleType), lit(0.0))
    }
    val withBathrooms = withColumnIfPresent(withTotalSf, "TotalBathrooms", Seq("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath")) {
      coalesce(col("FullBath").cast(DoubleType), lit(0.0)) +
        coalesce(col("HalfBath").cast(DoubleType), lit(0.0)) * lit(0.5) +
        coalesce(col("BsmtFullBath").cast(DoubleType), lit(0.0)) +
        coalesce(col("BsmtHalfBath").cast(DoubleType), lit(0.0)) * lit(0.5)
    }
    val withPorch = withColumnIfPresent(withBathrooms, "TotalPorchSF", Seq("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch")) {
      coalesce(col("OpenPorchSF").cast(DoubleType), lit(0.0)) +
        coalesce(col("EnclosedPorch").cast(DoubleType), lit(0.0)) +
        coalesce(col("3SsnPorch").cast(DoubleType), lit(0.0)) +
        coalesce(col("ScreenPorch").cast(DoubleType), lit(0.0))
    }
    val withQualityArea = withColumnIfPresent(withPorch, "OverallQualXTotalSF", Seq("OverallQual", "TotalSF")) {
      col("OverallQual").cast(DoubleType) * col("TotalSF").cast(DoubleType)
    }
    val withGarageAge = withColumnIfPresent(withQualityArea, "GarageAgeAtSale", Seq("YrSold", "GarageYrBlt")) {
      col("YrSold").cast(DoubleType) - col("GarageYrBlt").cast(DoubleType)
    }

    if (!advancedFeatures) {
      return addLogFeatures(
        withGarageAge,
        Seq("LotArea", "GrLivArea", "TotalSF", "TotalBsmtSF", "GarageArea", "MasVnrArea", "WoodDeckSF", "OpenPorchSF", "TotalPorchSF")
      )
    }

    val withQualitySquared = withColumnIfPresent(withGarageAge, "OverallQualSquared", Seq("OverallQual")) {
      pow(col("OverallQual").cast(DoubleType), 2.0)
    }
    val withQualityCubed = withColumnIfPresent(withQualitySquared, "OverallQualCubed", Seq("OverallQual")) {
      pow(col("OverallQual").cast(DoubleType), 3.0)
    }
    val withQualityLivingArea = withColumnIfPresent(withQualityCubed, "OverallQualXGrLivArea", Seq("OverallQual", "GrLivArea")) {
      col("OverallQual").cast(DoubleType) * col("GrLivArea").cast(DoubleType)
    }
    val withTotalHomeSF = withColumnIfPresent(withQualityLivingArea, "TotalHomeSF", Seq("BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF")) {
      Seq("BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF")
        .map(c => coalesce(col(c).cast(DoubleType), lit(0.0)))
        .reduce(_ + _)
    }
    val withPresenceFlags = Seq(
      "HasPool" -> "PoolArea",
      "HasSecondFloor" -> "2ndFlrSF",
      "HasGarage" -> "GarageArea",
      "HasBasement" -> "TotalBsmtSF",
      "HasFireplace" -> "Fireplaces"
    ).foldLeft(withTotalHomeSF) { case (acc, (feature, source)) =>
      withColumnIfPresent(acc, feature, Seq(source)) {
        when(coalesce(col(source).cast(DoubleType), lit(0.0)) > 0.0, 1.0).otherwise(0.0)
      }
    }

    addLogFeatures(
      withPresenceFlags,
      Seq(
        "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
        "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
        "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
        "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "TotalSF", "TotalHomeSF", "TotalPorchSF"
      )
    )
  }

  private def addLogFeatures(df: DataFrame, columns: Seq[String]): DataFrame =
    columns.filter(df.columns.contains).foldLeft(df) { case (acc, c) =>
      acc.withColumn(s"${c}Log1p", log1p(greatest(coalesce(col(c).cast(DoubleType), lit(0.0)), lit(0.0))))
    }

  private def withColumnIfPresent(df: DataFrame, name: String, required: Seq[String])(expr: => Column): DataFrame =
    if (required.forall(df.columns.contains)) df.withColumn(name, expr) else df

  private def withCategoricalText(df: DataFrame, categoricalColumns: Array[String]): DataFrame =
    categoricalColumns.foldLeft(df) { case (acc, c) =>
      acc.withColumn(textCol(c), coalesce(col(c).cast(StringType), lit("missing")))
    }

  private def filterTrainingOutliers(df: DataFrame, enabled: Boolean): DataFrame =
    if (enabled && Seq("GrLivArea", TargetCol).forall(df.columns.contains)) {
      df.filter(
        col("GrLivArea").cast(DoubleType) < 4000.0 ||
          col(TargetCol).cast(DoubleType) >= 300000.0 ||
          col("GrLivArea").isNull ||
          col(TargetCol).isNull
      )
    } else {
      df
    }

  private def writeSubmission(predictions: DataFrame, outputPath: String): Unit = {
    val submission = predictions
      .select(
        col(IdCol).cast(IntegerType).as(IdCol),
        greatest(exp(col("prediction")) - lit(1.0), lit(0.0)).as(TargetCol)
      )
      .orderBy(IdCol)

    writeSingleCsv(submission, outputPath)
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

  private def textCol(column: String): String =
    s"${column}_text"

  private def isNumeric(dataType: org.apache.spark.sql.types.DataType): Boolean =
    dataType match {
      case _: NumericType => true
      case _ => false
    }

  private def parseArgs(args: List[String], config: Config): Config =
    args match {
      case Nil => config
      case "--train" :: value :: tail => parseArgs(tail, config.copy(trainPath = value))
      case "--test" :: value :: tail => parseArgs(tail, config.copy(testPath = value))
      case "--output" :: value :: tail => parseArgs(tail, config.copy(outputPath = value))
      case "--model" :: value :: tail => parseArgs(tail, config.copy(modelPath = value))
      case "--algo" :: value :: tail => parseArgs(tail, config.copy(algo = value))
      case "--max-bins" :: value :: tail => parseArgs(tail, config.copy(maxBins = value.toInt))
      case "--num-trees" :: value :: tail => parseArgs(tail, config.copy(numTrees = value.toInt))
      case "--max-depth" :: value :: tail => parseArgs(tail, config.copy(maxDepth = value.toInt))
      case "--max-iter" :: value :: tail => parseArgs(tail, config.copy(maxIter = value.toInt))
      case "--reg-param" :: value :: tail => parseArgs(tail, config.copy(regParam = value.toDouble))
      case "--elastic-net" :: value :: tail => parseArgs(tail, config.copy(elasticNetParam = value.toDouble))
      case "--xgb-rounds" :: value :: tail => parseArgs(tail, config.copy(xgbRounds = value.toInt))
      case "--xgb-eta" :: value :: tail => parseArgs(tail, config.copy(xgbEta = value.toDouble))
      case "--ensemble-xgb-weight" :: value :: tail => parseArgs(tail, config.copy(ensembleXgbWeight = value.toDouble))
      case "--advanced-features" :: tail => parseArgs(tail, config.copy(advancedFeatures = true))
      case "--basic-features" :: tail => parseArgs(tail, config.copy(advancedFeatures = false))
      case "--remove-outliers" :: tail => parseArgs(tail, config.copy(removeOutliers = true))
      case "--keep-outliers" :: tail => parseArgs(tail, config.copy(removeOutliers = false))
      case "--validation-only" :: tail => parseArgs(tail, config.copy(validationOnly = true))
      case "--validation-seeds" :: value :: tail =>
        parseArgs(tail, config.copy(validationSeeds = value.split(",").filter(_.nonEmpty).map(_.toLong).toSeq))
      case "--seed" :: value :: tail => parseArgs(tail, config.copy(seed = value.toLong))
      case "--help" :: _ =>
        println(
          s"""
             |Usage:
             |  housePrices/run [--algo lr|rf|gbt|xgb|ensemble] [--validation-only] [--validation-seeds 42,1337,2026] [--advanced-features|--basic-features] [--remove-outliers|--keep-outliers] [--max-bins n] [--num-trees n] [--max-depth n] [--max-iter n] [--reg-param x] [--elastic-net x] [--xgb-rounds n] [--xgb-eta x] [--ensemble-xgb-weight x] [--train path] [--test path] [--output path] [--model path] [--seed n]
             |
             |Defaults:
             |  --algo ${config.algo}
             |  --validation-only ${config.validationOnly}
             |  --max-bins ${config.maxBins}
             |  --num-trees ${config.numTrees}
             |  --max-depth ${config.maxDepth}
             |  --max-iter ${config.maxIter}
             |  --reg-param ${config.regParam}
             |  --elastic-net ${config.elasticNetParam}
             |  --xgb-rounds ${config.xgbRounds}
             |  --xgb-eta ${config.xgbEta}
             |  --ensemble-xgb-weight ${config.ensembleXgbWeight}
             |  --advanced-features ${config.advancedFeatures}
             |  --keep-outliers ${!config.removeOutliers}
             |  --train ${config.trainPath}
             |  --test ${config.testPath}
             |  --output ${config.outputPath}
             |  --model ${config.modelPath}
             |""".stripMargin.trim
        )
        sys.exit(0)
      case flag :: _ if flag.startsWith("--") =>
        throw new IllegalArgumentException(s"Unknown argument: $flag")
      case value :: _ =>
        throw new IllegalArgumentException(s"Unexpected positional argument: $value")
    }
}
