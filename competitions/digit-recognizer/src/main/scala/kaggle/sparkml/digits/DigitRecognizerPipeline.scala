package kaggle.sparkml.digits

import kaggle.sparkml.core.{BatchClassification, BatchConfig, ClassificationCompetition}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{ElementwiseProduct, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}

object DigitRecognizerPipeline extends ClassificationCompetition {
  override val name = "digit-recognizer"
  override val defaultModel = "models/multinomial_lr_v1"
  override val tuningOptions = Set("--max-iter", "--reg-param")
  override val tuningHelp = "--max-iter n (100) --reg-param x (0.01)"
  private val Pixels = (0 until 784).map(i => s"pixel$i").toArray

  def main(args: Array[String]): Unit = BatchClassification.main(args, this)

  override def read(spark: SparkSession, path: String, training: Boolean): DataFrame = {
    val input = new Path(path)
    val fs = input.getFileSystem(spark.sparkContext.hadoopConfiguration)
    require(fs.exists(input) && fs.getFileStatus(input).isFile,
      "Digit Recognizer input must be one CSV file, not a directory or glob")
    val lines = spark.sparkContext.textFile(path)
    val columns = if (training) Array("label") ++ Pixels else Pixels
    require(lines.take(1).headOption.exists(_.stripPrefix("\uFEFF").split(",", -1).sameElements(columns)),
      s"Expected CSV header: ${columns.mkString(",")}")
    // zipWithIndex runs before parsing/repartitioning: ImageId is the 1-based physical CSV row.
    val numbered = lines.zipWithIndex().filter(_._2 > 0).map { case (line, index) => Row(index, line) }
    val raw = spark.createDataFrame(numbered, StructType(Seq(
      StructField("ImageId", LongType, nullable = false), StructField("csv", StringType, nullable = false)
    )))
    require(raw.take(1).nonEmpty, "Input CSV must contain at least one image")
    require(raw.filter(size(split(col("csv"), ",", -1)) =!= columns.length).take(1).isEmpty,
      s"Each image row must contain exactly ${columns.length} numeric fields")
    // Decode each numeric row once. Expanding from_csv into 784 field expressions
    // can instantiate hundreds of parser buffers after Catalyst projection pruning.
    val parsed = numbered.map { row =>
      val values = row.getString(1).split(",", -1).map { value =>
        scala.util.Try(value.toDouble).getOrElse(Double.NaN)
      }
      Row.fromSeq(Seq(row.getLong(0)) ++ values.toSeq)
    }
    val schema = StructType(Seq(StructField("ImageId", LongType, nullable = false)) ++
      columns.map(name => StructField(name, DoubleType, nullable = false)))
    val frame = spark.createDataFrame(parsed, schema)
    val pixelArray = Pixels.mkString("array(", ",", ")")
    require(frame.filter(expr(s"exists($pixelArray, x -> x IS NULL OR isnan(x) OR x < 0 OR x > 255 OR x != floor(x))"))
      .take(1).isEmpty, "Pixels must be integers in [0, 255] without missing values")
    if (training) {
      require(frame.filter(col("label").isNull || isnan(col("label")) || col("label") < 0 ||
        col("label") > 9 || col("label") =!= floor(col("label"))).take(1).isEmpty,
        "Digit labels must be integers in [0, 9]")
    }
    frame
  }

  override def pipeline(config: BatchConfig): Pipeline = {
    val iterations = config.intOption("--max-iter", 100)
    val regularization = config.doubleOption("--reg-param", 0.01)
    require(iterations > 0, "--max-iter must be positive")
    require(!regularization.isNaN && !regularization.isInfinity && regularization >= 0,
      "--reg-param must be finite and non-negative")
    val assembler = new VectorAssembler().setInputCols(Pixels).setOutputCol("rawFeatures")
    val scale = new ElementwiseProduct().setInputCol("rawFeatures").setOutputCol("features")
      .setScalingVec(Vectors.dense(Array.fill(784)(1.0 / 255.0)))
    val classifier = new LogisticRegression().setFamily("multinomial").setMaxIter(iterations)
      .setRegParam(regularization).setLabelCol("label").setFeaturesCol("features")
    new Pipeline().setStages(Array(assembler, scale, classifier))
  }

  override def submission(predictions: DataFrame): DataFrame = predictions.select(
    col("ImageId"), col("prediction").cast("int").as("Label")
  ).orderBy("ImageId")
}
