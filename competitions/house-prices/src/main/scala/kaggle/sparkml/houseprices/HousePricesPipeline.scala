package kaggle.sparkml.houseprices

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, NumericType, StringType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

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
      validationOnly: Boolean = false,
      seed: Long = 42L
  )

  def main(args: Array[String]): Unit = {
    val config = parseArgs(args.toList, Config())
    require(Set("lr", "rf", "gbt").contains(config.algo), s"Unknown --algo: ${config.algo}")
    require(config.maxBins >= 32, "--max-bins must be at least 32")

    val spark = SparkSession
      .builder()
      .appName("kaggle-house-prices-spark")
      .master(sys.props.getOrElse("spark.master", "local[*]"))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      val rawTrain = withHouseFeatures(readCsv(spark, config.trainPath))
      val rawTest = withHouseFeatures(readCsv(spark, config.testPath))
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
      println(s"Validation only: ${config.validationOnly}")

      val preparedTrain = withCategoricalText(train, categoricalColumns)
      val preparedTest = withCategoricalText(rawTest, categoricalColumns)
      val Array(training, validation) = preparedTrain.randomSplit(Array(0.8, 0.2), config.seed)

      val pipeline = buildPipeline(numericColumns, categoricalColumns, config)
      val validationModel = pipeline.fit(training)
      val validationPredictions = validationModel.transform(validation)
      val logRmse = new RegressionEvaluator()
        .setLabelCol(LabelCol)
        .setPredictionCol("prediction")
        .setMetricName("rmse")
        .evaluate(validationPredictions)

      println(f"Validation log-RMSE: $logRmse%.5f")

      if (!config.validationOnly) {
        val finalModel = pipeline.fit(preparedTrain)
        finalModel.write.overwrite().save(config.modelPath)
        writeSubmission(finalModel.transform(preparedTest), config.outputPath)
        println(s"Model saved: ${config.modelPath}")
        println(s"Submission saved: ${config.outputPath}")
      }
    } finally {
      spark.stop()
    }
  }

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
    }

    new Pipeline().setStages(stages.toArray)
  }

  private def readCsv(spark: SparkSession, path: String): DataFrame =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("nullValue", "NA")
      .csv(path)

  private def withHouseFeatures(df: DataFrame): DataFrame = {
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

    addLogFeatures(withGarageAge, Seq("LotArea", "GrLivArea", "TotalSF", "TotalBsmtSF", "GarageArea", "MasVnrArea", "WoodDeckSF", "OpenPorchSF", "TotalPorchSF"))
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
      case "--validation-only" :: tail => parseArgs(tail, config.copy(validationOnly = true))
      case "--seed" :: value :: tail => parseArgs(tail, config.copy(seed = value.toLong))
      case "--help" :: _ =>
        println(
          s"""
             |Usage:
             |  housePrices/run [--algo lr|rf|gbt] [--validation-only] [--max-bins n] [--num-trees n] [--max-depth n] [--max-iter n] [--reg-param x] [--elastic-net x] [--train path] [--test path] [--output path] [--model path] [--seed n]
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
