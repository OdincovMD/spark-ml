package kaggle.sparkml.disasters

import kaggle.sparkml.core.{BatchClassification, BatchConfig, ClassificationCompetition}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, NGram, Normalizer, RegexTokenizer, SQLTransformer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, LongType}

object DisasterTweetsPipeline extends ClassificationCompetition {
  override val name = "disaster-tweets"
  override val defaultModel = "models/tfidf_lr_v3"
  override val tuningOptions = Set("--num-features", "--max-iter", "--reg-param", "--threshold")
  override val tuningHelp = "--num-features n (32768) --max-iter n (80) --reg-param x (0.0001) --threshold x (0.4)"

  private val Fields = Array("id", "keyword", "location", "text")

  def main(args: Array[String]): Unit = BatchClassification.main(args, this)

  override def read(spark: SparkSession, path: String, training: Boolean): DataFrame = {
    val columns = if (training) Fields :+ "target" else Fields
    // Tweets contain commas, escaped quotes and line breaks; inference must use the same CSV parser.
    val source = spark.read.option("header", "true").option("inferSchema", "false")
      .option("multiLine", "true").option("escape", "\"").option("mode", "FAILFAST").csv(path)
    require(source.columns.sameElements(columns), s"Expected CSV header: ${columns.mkString(",")}")
    require(source.take(1).nonEmpty, "Input CSV must contain at least one tweet")
    val frame = source.withColumn("tweetId", col("id").cast(LongType))
    require(frame.filter(col("id").isNull || !col("id").rlike("^[0-9]+$") || col("tweetId").isNull)
      .take(1).isEmpty, "id must be a nonnegative integer within the Long range")
    require(frame.groupBy("tweetId").count().filter(col("count") > 1).take(1).isEmpty,
      "id must be unique")
    val typed = frame.drop("id").withColumnRenamed("tweetId", "id")
    if (training) {
      require(typed.filter(col("target").isNull || !col("target").isin("0", "1"))
        .take(1).isEmpty, "target must contain 0 or 1")
      typed.withColumn("label", col("target").cast("double")).drop("target")
    } else typed
  }

  override def pipeline(config: BatchConfig): Pipeline = {
    val features = config.intOption("--num-features", 32768)
    val iterations = config.intOption("--max-iter", 80)
    val regularization = config.doubleOption("--reg-param", 0.0001)
    val threshold = config.doubleOption("--threshold", 0.4)
    require(features > 0, "--num-features must be positive")
    require(iterations > 0, "--max-iter must be positive")
    require(!regularization.isInfinity && !regularization.isNaN && regularization >= 0.0,
      "--reg-param must be finite and nonnegative")
    require(threshold > 0.0 && threshold < 1.0, "--threshold must be in (0, 1)")
    val clean = new SQLTransformer().setStatement(
      """SELECT *, concat(
        |  regexp_replace(regexp_replace(COALESCE(text, ''), 'https?://[^\\s]+', ' '),
        |    '@[A-Za-z0-9_]+', ' '), ' ',
        |  regexp_replace(COALESCE(keyword, ''), '%20', ' ')) AS cleanText
        |FROM __THIS__""".stripMargin)
    val tokenizer = new RegexTokenizer().setInputCol("cleanText").setOutputCol("words")
      .setPattern("[^\\p{L}\\p{N}_]+").setToLowercase(true).setMinTokenLength(2)
    val bigrams = new NGram().setN(2).setInputCol("words").setOutputCol("bigrams")
    val terms = new SQLTransformer().setStatement("SELECT *, concat(words, bigrams) AS terms FROM __THIS__")
    val hashing = new HashingTF().setNumFeatures(features).setInputCol("terms").setOutputCol("termCounts")
    val idf = new IDF().setMinDocFreq(2).setInputCol("termCounts").setOutputCol("tfidf")
    val normalize = new Normalizer().setInputCol("tfidf").setOutputCol("features").setP(2.0)
    val classifier = new LogisticRegression().setMaxIter(iterations).setRegParam(regularization)
      .setElasticNetParam(0.0).setStandardization(false).setThreshold(threshold)
      .setLabelCol("label").setFeaturesCol("features")
    new Pipeline().setStages(Array(clean, tokenizer, bigrams, terms, hashing, idf, normalize, classifier))
  }

  override def validationMetrics(predictions: DataFrame): Seq[(String, Double)] = {
    val counts = predictions.agg(
      sum(when(col("label") === 1.0 && col("prediction") === 1.0, 1L).otherwise(0L)).as("tp"),
      sum(when(col("label") === 0.0 && col("prediction") === 1.0, 1L).otherwise(0L)).as("fp"),
      sum(when(col("label") === 1.0 && col("prediction") === 0.0, 1L).otherwise(0L)).as("fn")
    ).head()
    val tp = counts.getLong(0).toDouble
    val denominator = 2.0 * tp + counts.getLong(1) + counts.getLong(2)
    Seq("F1 (target=1)" -> (if (denominator == 0.0) 0.0 else 2.0 * tp / denominator))
  }

  override def submission(predictions: DataFrame): DataFrame = predictions.select(
    col("id"), col("prediction").cast(IntegerType).as("target")
  ).orderBy("id")
}
