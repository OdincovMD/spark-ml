package kaggle.sparkml.titanic

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

object TitanicPipeline {
  final case class Config(
      trainPath: String = "data/raw/train.csv",
      testPath: String = "data/raw/test.csv",
      outputPath: String = "output/submission.csv",
      modelPath: String = "models/random_forest_pipeline",
      seed: Long = 42L
  )

  def main(args: Array[String]): Unit = {
    val config = parseArgs(args.toList, Config())

    val spark = SparkSession
      .builder()
      .appName("kaggle-titanic-spark")
      .master(sys.props.getOrElse("spark.master", "local[*]"))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      val train = readCsv(spark, config.trainPath)
      val test = readCsv(spark, config.testPath)

      val Array(training, validation) = train.randomSplit(Array(0.8, 0.2), config.seed)
      val pipeline = buildPipeline(config.seed)
      val model = pipeline.fit(training)

      val validationPredictions = model.transform(validation)
      val auc = new BinaryClassificationEvaluator()
        .setLabelCol("Survived")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")
        .evaluate(validationPredictions)

      val accuracy = validationPredictions
        .select((col("prediction") === col("Survived").cast("double")).cast("double").as("ok"))
        .agg(org.apache.spark.sql.functions.avg("ok").as("accuracy"))
        .first()
        .getAs[Double]("accuracy")

      println(f"Validation accuracy: $accuracy%.4f")
      println(f"Validation AUC:      $auc%.4f")

      val fullModel = pipeline.fit(train)
      saveModel(fullModel, config.modelPath)

      val submission = fullModel
        .transform(test)
        .select(
          col("PassengerId").cast(IntegerType).as("PassengerId"),
          col("prediction").cast(IntegerType).as("Survived")
        )
        .orderBy("PassengerId")

      writeSingleCsv(submission, config.outputPath)
      println(s"Wrote Kaggle submission to ${config.outputPath}")
    } finally {
      spark.stop()
    }
  }

  private def readCsv(spark: SparkSession, path: String): DataFrame =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)

  private def buildPipeline(seed: Long): Pipeline = {
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
      new StringIndexer()
        .setInputCol("Sex")
        .setOutputCol("SexIndexed")
        .setHandleInvalid("keep"),
      new StringIndexer()
        .setInputCol("EmbarkedText")
        .setOutputCol("EmbarkedIndexed")
        .setHandleInvalid("keep"),
      new StringIndexer()
        .setInputCol("PclassText")
        .setOutputCol("PclassIndexed")
        .setHandleInvalid("keep"),
      new StringIndexer()
        .setInputCol("Title")
        .setOutputCol("TitleIndexed")
        .setHandleInvalid("keep")
    )

    val assembler = new VectorAssembler()
      .setInputCols(
        Array(
          "PclassIndexed",
          "SexIndexed",
          "EmbarkedIndexed",
          "TitleIndexed",
          "AgeImputed",
          "FareImputed",
          "SibSp",
          "Parch",
          "FamilySize",
          "IsAlone",
          "FarePerPerson"
        )
      )
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val classifier = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setSeed(seed)
      .setNumTrees(200)
      .setMaxDepth(6)

    new Pipeline()
      .setStages(Array[PipelineStage](baseFeatures, imputer, derivedFeatures) ++ indexers ++ Array(assembler, classifier))
  }

  private def sqlTransformer(statement: String): org.apache.spark.ml.feature.SQLTransformer =
    new org.apache.spark.ml.feature.SQLTransformer().setStatement(statement)

  private def saveModel(model: PipelineModel, path: String): Unit = {
    model.write.overwrite().save(path)
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

  private def parseArgs(args: List[String], config: Config): Config =
    args match {
      case Nil => config
      case "--train" :: value :: tail => parseArgs(tail, config.copy(trainPath = value))
      case "--test" :: value :: tail => parseArgs(tail, config.copy(testPath = value))
      case "--output" :: value :: tail => parseArgs(tail, config.copy(outputPath = value))
      case "--model" :: value :: tail => parseArgs(tail, config.copy(modelPath = value))
      case "--seed" :: value :: tail => parseArgs(tail, config.copy(seed = value.toLong))
      case "--help" :: _ =>
        println(
          """
            |Usage:
            |  titanic/run [--train path] [--test path] [--output path] [--model path] [--seed n]
            |""".stripMargin.trim
        )
        sys.exit(0)
      case flag :: _ if flag.startsWith("--") =>
        throw new IllegalArgumentException(s"Unknown argument: $flag")
      case value :: _ =>
        throw new IllegalArgumentException(s"Unexpected positional argument: $value")
    }
}
