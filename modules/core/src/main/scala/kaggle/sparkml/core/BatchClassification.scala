package kaggle.sparkml.core

import java.util.UUID

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

final case class BatchConfig(
    mode: String = "run",
    trainPath: String = "data/raw/train.csv",
    inputPath: String = "data/raw/test.csv",
    outputPath: String = "output/submission.csv",
    modelPath: String = "models/baseline",
    seed: Long = 42L,
    validationFraction: Double = 0.2,
    overwrite: Boolean = false,
    tuning: Map[String, String] = Map.empty
) {
  def intOption(name: String, default: Int): Int = tuning.get(name).map(_.toInt).getOrElse(default)
  def doubleOption(name: String, default: Double): Double = tuning.get(name).map(_.toDouble).getOrElse(default)
}

trait ClassificationCompetition {
  def name: String
  def defaultModel: String
  def tuningOptions: Set[String]
  def tuningHelp: String
  def read(spark: SparkSession, path: String, training: Boolean): DataFrame
  def pipeline(config: BatchConfig): Pipeline
  def submission(predictions: DataFrame): DataFrame
  def validationMetrics(predictions: DataFrame): Seq[(String, Double)] = Seq.empty
}

object BatchClassification {
  def main(args: Array[String], competition: ClassificationCompetition): Unit = {
    if (args.contains("--help")) {
      println(s"""${competition.name}: train | predict | run (default)
                 |  --train path                Labeled training CSV
                 |  --input path                Unlabeled inference CSV
                 |  --model path                Saved Spark PipelineModel
                 |  --output path               Single submission CSV
                 |  --seed n                    Training/holdout seed (42)
                 |  --validation-fraction x     Holdout fraction (0.2)
                 |  --overwrite                 Replace existing model/output
                 |  ${competition.tuningHelp}
                 |predict reads only --input and --model; it never fits a model.
                 |Paths are relative to the competition directory under sbt run.
                 |""".stripMargin)
      return
    }
    val config = parse(args.toList, competition.defaultModel, competition.tuningOptions)
    val spark = SparkSession.builder()
      .appName(s"kaggle-${competition.name}-${config.mode}")
      .master(sys.props.getOrElse("spark.master", "local[*]"))
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try execute(spark, config, competition)
    finally spark.stop()
  }

  def parse(args: List[String], defaultModel: String, tuningOptions: Set[String]): BatchConfig = {
    val modes = Set("train", "predict", "run")
    val (mode, options) = args match {
      case head :: tail if modes.contains(head) => (head, tail)
      case _ => ("run", args)
    }
    @annotation.tailrec
    def loop(rest: List[String], config: BatchConfig): BatchConfig = rest match {
      case Nil => config
      case "--overwrite" :: tail => loop(tail, config.copy(overwrite = true))
      case option :: value :: tail if !value.startsWith("--") =>
        val next = option match {
          case "--train" => config.copy(trainPath = value)
          case "--input" => config.copy(inputPath = value)
          case "--output" => config.copy(outputPath = value)
          case "--model" => config.copy(modelPath = value)
          case "--seed" => config.copy(seed = value.toLong)
          case "--validation-fraction" => config.copy(validationFraction = value.toDouble)
          case tuning if tuningOptions.contains(tuning) => config.copy(tuning = config.tuning + (tuning -> value))
          case _ => throw new IllegalArgumentException(s"Unknown option: $option; use --help")
        }
        require(value.trim.nonEmpty, s"Empty value for $option")
        loop(tail, next)
      case _ => throw new IllegalArgumentException(s"Invalid arguments: ${rest.mkString(" ")}; use --help")
    }
    val config = loop(options, BatchConfig(mode = mode, modelPath = defaultModel))
    require(config.validationFraction > 0.0 && config.validationFraction < 1.0,
      "--validation-fraction must be in (0, 1)")
    config
  }

  def execute(spark: SparkSession, config: BatchConfig, competition: ClassificationCompetition): Unit = {
    if (config.mode != "train") checkDestination(spark, config.outputPath, config.overwrite, directory = false)
    if (config.mode != "predict") {
      checkDestination(spark, config.modelPath, config.overwrite, directory = true)
      val train = competition.read(spark, config.trainPath, training = true).cache()
      try {
        val Array(training, validation) = train.randomSplit(
          Array(1.0 - config.validationFraction, config.validationFraction), config.seed)
        require(training.take(1).nonEmpty && validation.take(1).nonEmpty,
          "Empty training/validation split; provide more rows or adjust --validation-fraction")
        require(training.select("label").distinct().count() >= 2,
          "The training split must contain at least two classes")
        val pipeline = competition.pipeline(config)
        val validationModel = pipeline.fit(training)
        val predictions = validationModel.transform(validation).cache()
        try {
          val accuracy = new MulticlassClassificationEvaluator()
            .setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
            .evaluate(predictions)
          println(f"Validation accuracy: $accuracy%.5f (seed=${config.seed}, holdout=${config.validationFraction})")
          competition.validationMetrics(predictions).foreach { case (metric, value) =>
            println(f"Validation $metric: $value%.5f (seed=${config.seed}, holdout=${config.validationFraction})")
          }
        } finally predictions.unpersist()
        val fitted = pipeline.fit(train)
        val writer = fitted.write
        if (config.overwrite) writer.overwrite().save(config.modelPath)
        else writer.save(config.modelPath)
        println(s"Saved full pipeline: ${config.modelPath}")
      } finally train.unpersist()
    }
    if (config.mode != "train") {
      val model = PipelineModel.load(config.modelPath)
      val input = competition.read(spark, config.inputPath, training = false)
      val submission = competition.submission(model.transform(input))
      writeSingleCsv(submission, config.outputPath, config.overwrite)
      println(s"Submission: ${config.outputPath}")
    }
  }

  private def checkDestination(spark: SparkSession, path: String, overwrite: Boolean, directory: Boolean): Unit = {
    val target = new Path(path)
    val fs = target.getFileSystem(spark.sparkContext.hadoopConfiguration)
    if (fs.exists(target)) {
      require(overwrite, s"Path already exists: $path; use a new path or --overwrite")
      require(fs.getFileStatus(target).isDirectory == directory,
        s"Expected ${if (directory) "a model directory" else "a CSV file"}: $path")
      if (directory) {
        require(fs.exists(new Path(target, "metadata")) && fs.exists(new Path(target, "stages")),
          s"Refusing to overwrite a directory without Spark PipelineModel metadata: $path")
      }
    }
  }

  def writeSingleCsv(df: DataFrame, outputPath: String, overwrite: Boolean): Unit = {
    val spark = df.sparkSession
    checkDestination(spark, outputPath, overwrite, directory = false)
    val output = new Path(outputPath)
    val fs = output.getFileSystem(spark.sparkContext.hadoopConfiguration)
    val temporary = new Path(s"$outputPath.spark-tmp-${UUID.randomUUID()}")
    val backup = new Path(s"$outputPath.backup-${UUID.randomUUID()}")
    var backedUp = false
    try {
      df.coalesce(1).write.option("header", "true").csv(temporary.toString)
      val parts = fs.globStatus(new Path(temporary, "part-*.csv"))
      require(parts != null && parts.length == 1, s"Expected one CSV part under $temporary")
      if (fs.exists(output)) {
        require(overwrite, s"Output already exists: $outputPath")
        require(fs.rename(output, backup), s"Could not back up $output")
        backedUp = true
      }
      require(fs.rename(parts.head.getPath, output), s"Could not move CSV into $output")
      if (backedUp) fs.delete(backup, false)
    } catch {
      case error: Exception =>
        if (backedUp && !fs.exists(output)) {
          try {
            if (!fs.rename(backup, output)) {
              error.addSuppressed(new IllegalStateException(s"Previous CSV could not be restored; it is at $backup"))
            }
          } catch { case restoreError: Exception => error.addSuppressed(restoreError) }
        }
        throw error
    } finally fs.delete(temporary, true)
  }
}
