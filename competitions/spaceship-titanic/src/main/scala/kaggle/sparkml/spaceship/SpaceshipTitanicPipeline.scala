package kaggle.sparkml.spaceship

import kaggle.sparkml.core.{BatchClassification, BatchConfig, ClassificationCompetition}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{Imputer, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object SpaceshipTitanicPipeline extends ClassificationCompetition {
  override val name = "spaceship-titanic"
  override val defaultModel = "models/rf_v1"
  override val tuningOptions = Set("--classifier", "--num-trees", "--max-depth", "--max-iter", "--step-size", "--feature-set")
  override val tuningHelp = "--classifier rf|gbt (rf) --num-trees n (200) --max-depth n (8) " +
    "--max-iter n (100, GBT) --step-size x (0.05, GBT) --feature-set basic|extended (basic)"

  private val Spending = Array("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
  private val Numeric = Array("Age", "CabinNumber", "GroupMember", "TotalSpend", "NoSpend") ++ Spending
  private val ExtendedNumeric = Array("GroupNumber", "MissingSpendCount", "SpendServiceCount") ++
    Spending.map(name => s"${name}Share")
  private val Categorical = Array("HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide")
  private val Fields = Array("PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP") ++
    Spending ++ Array("Name")

  def main(args: Array[String]): Unit = BatchClassification.main(args, this)

  override def read(spark: SparkSession, path: String, training: Boolean): DataFrame = {
    val columns = if (training) Fields :+ "Transported" else Fields
    val source = spark.read.option("header", "true").option("inferSchema", "false")
      .option("mode", "FAILFAST").csv(path)
    val missing = columns.filterNot(source.columns.contains)
    require(missing.isEmpty, s"Missing CSV columns: ${missing.mkString(", ")}")
    val schema = StructType(columns.map { name =>
      StructField(name, if (name == "Age" || Spending.contains(name)) DoubleType else StringType, nullable = true)
    })
    // Match the header first: positional CSV schemas must not silently shuffle features.
    require(source.columns.sameElements(columns), s"Expected CSV header: ${columns.mkString(",")}")
    val frame = spark.read.schema(schema).option("header", "true").option("enforceSchema", "false")
      .option("mode", "FAILFAST").csv(path)
    require(frame.take(1).nonEmpty, "Input CSV must contain at least one passenger")
    require(frame.filter(col("PassengerId").isNull || !col("PassengerId").rlike("^[0-9]{4}_[0-9]{2}$"))
      .take(1).isEmpty, "PassengerId must use the gggg_pp format")
    require(frame.groupBy("PassengerId").count().filter(col("count") > 1).take(1).isEmpty,
      "PassengerId must be unique")
    if (training) {
      require(frame.filter(col("Transported").isNull || !lower(col("Transported")).isin("true", "false"))
        .take(1).isEmpty, "Transported must contain True or False")
      frame.withColumn("label", when(lower(col("Transported")) === "true", 1.0).otherwise(0.0))
        .drop("Transported")
    } else frame
  }

  override def pipeline(config: BatchConfig): Pipeline = {
    val trees = config.intOption("--num-trees", 200)
    val depth = config.intOption("--max-depth", 8)
    val algorithm = config.tuning.getOrElse("--classifier", "rf")
    val iterations = config.intOption("--max-iter", 100)
    val stepSize = config.doubleOption("--step-size", 0.05)
    val featureSet = config.tuning.getOrElse("--feature-set", "basic")
    require(Set("rf", "gbt").contains(algorithm), "--classifier must be rf or gbt")
    require(Set("basic", "extended").contains(featureSet), "--feature-set must be basic or extended")
    require(trees > 0, "--num-trees must be positive")
    require(depth > 0 && depth <= 30, "--max-depth must be in [1, 30]")
    require(iterations > 0, "--max-iter must be positive")
    require(stepSize > 0.0 && stepSize <= 1.0, "--step-size must be in (0, 1]")
    val spend = Spending.map(name => s"COALESCE($name, 0.0)").mkString(" + ")
    // Row-local features work identically on holdout and new batches, without group aggregation.
    val extended = if (featureSet == "extended") {
      val missing = Spending.map(name => s"CASE WHEN $name IS NULL THEN 1.0 ELSE 0.0 END").mkString(" + ")
      val services = Spending.map(name => s"CASE WHEN $name > 0 THEN 1.0 ELSE 0.0 END").mkString(" + ")
      val shares = Spending.map { name =>
        s"CASE WHEN ($spend) > 0 THEN COALESCE($name, 0.0) / ($spend) ELSE 0.0 END AS ${name}Share"
      }
      ", " + (Array("CAST(split(PassengerId, '_')[0] AS DOUBLE) AS GroupNumber",
        s"$missing AS MissingSpendCount", s"$services AS SpendServiceCount") ++ shares).mkString(", ")
    } else ""
    val numeric = if (featureSet == "extended") Numeric ++ ExtendedNumeric else Numeric
    val text = Array("HomePlanet", "CryoSleep", "Destination", "VIP")
      .map(name => s"COALESCE($name, '__missing__') AS ${name}Text")
    val featureStage = new SQLTransformer().setStatement(
      s"""SELECT *, ${text.mkString(", ")},
         |  COALESCE(split(Cabin, '/')[0], '__missing__') AS CabinDeck,
         |  COALESCE(split(Cabin, '/')[2], '__missing__') AS CabinSide,
         |  CAST(split(Cabin, '/')[1] AS DOUBLE) AS CabinNumber,
         |  CAST(split(PassengerId, '_')[1] AS DOUBLE) AS GroupMember,
         |  $spend AS TotalSpend,
         |  CASE WHEN $spend = 0 THEN 1.0 ELSE 0.0 END AS NoSpend$extended
         |FROM __THIS__""".stripMargin)
    val categorical = Categorical.map {
      case name if Array("HomePlanet", "CryoSleep", "Destination", "VIP").contains(name) => s"${name}Text"
      case name => name
    }
    val indexers = categorical.map { name =>
      new StringIndexer().setInputCol(name).setOutputCol(s"${name}Index").setHandleInvalid("keep")
    }
    val imputer = new Imputer().setStrategy("median").setInputCols(numeric)
      .setOutputCols(numeric.map(name => s"${name}Imputed"))
    val assembler = new VectorAssembler()
      .setInputCols(numeric.map(name => s"${name}Imputed") ++ categorical.map(name => s"${name}Index"))
      .setOutputCol("features")
    val classifier: PipelineStage = algorithm match {
      case "rf" => new RandomForestClassifier().setNumTrees(trees).setMaxDepth(depth)
        .setMaxBins(64).setSeed(config.seed).setLabelCol("label").setFeaturesCol("features")
      case "gbt" => new GBTClassifier().setMaxIter(iterations).setStepSize(stepSize).setMaxDepth(depth)
        .setMaxBins(64).setSeed(config.seed).setLabelCol("label").setFeaturesCol("features")
    }
    new Pipeline().setStages(Array[PipelineStage](featureStage, imputer) ++ indexers ++ Array(assembler, classifier))
  }

  override def submission(predictions: DataFrame): DataFrame = predictions.select(
    col("PassengerId"), when(col("prediction") === 1.0, "True").otherwise("False").as("Transported")
  ).orderBy("PassengerId")
}
