package kaggle.sparkml.digits

import java.nio.file.Files

import kaggle.sparkml.core.{BatchClassification, BatchConfig, SparkSuite}
import org.apache.spark.ml.PipelineModel

class DigitRecognizerPipelineSpec extends SparkSuite {
  private val header = (0 until 784).map(i => s"pixel$i").mkString(",")
  private def image(digit: Int): String = (0 until 784).map(i => if (i == digit) "255" else "0").mkString(",")

  test("saved multinomial pipeline predicts unlabeled CSV in physical row order") {
    val dir = Files.createTempDirectory("digits-pipeline-")
    val training = writeCsv(dir.resolve("train.csv"), Seq("label," + header) ++ (0 until 100).map { i =>
      s"${i % 10},${image(i % 10)}"
    })
    val digits = Vector(9, 0, 5, 1)
    val input = writeCsv(dir.resolve("test.csv"), Seq(header) ++ digits.map(image))
    val modelPath = dir.resolve("model").toString
    BatchClassification.execute(spark, BatchConfig(mode = "train", trainPath = training,
      modelPath = modelPath, tuning = Map("--max-iter" -> "10")), DigitRecognizerPipeline)
    val frame = DigitRecognizerPipeline.read(spark, input, training = false)
    assert(!frame.columns.contains("label"))
    assert(frame.orderBy("ImageId").select("ImageId").collect().map(_.getLong(0)).toVector == Vector(1L, 2L, 3L, 4L))
    val loaded = PipelineModel.load(modelPath)
    val expected = DigitRecognizerPipeline.submission(loaded.transform(frame)).collect()
      .map(row => s"${row.getLong(0)},${row.getInt(1)}").toVector
    assert(expected == digits.zipWithIndex.map { case (digit, index) => s"${index + 1},$digit" })
    val output = dir.resolve("submission.csv")
    BatchClassification.execute(spark, BatchConfig(mode = "predict", trainPath = "does-not-exist.csv",
      inputPath = input, outputPath = output.toString, modelPath = modelPath), DigitRecognizerPipeline)
    assert(readLines(output) == Vector("ImageId,Label") ++ expected)
  }

  test("pixel header order, missing or out-of-range pixels and invalid labels are rejected") {
    val dir = Files.createTempDirectory("digits-invalid-")
    val wrongOrder = writeCsv(dir.resolve("header.csv"), Seq(header.replace("pixel0,pixel1,", "pixel1,pixel0,"), image(0)))
    intercept[IllegalArgumentException] { DigitRecognizerPipeline.read(spark, wrongOrder, training = false) }
    val outOfRange = writeCsv(dir.resolve("range.csv"), Seq(header, image(0).replaceFirst("255", "256")))
    intercept[IllegalArgumentException] { DigitRecognizerPipeline.read(spark, outOfRange, training = false) }
    val missing = writeCsv(dir.resolve("missing.csv"), Seq(header, image(0).replaceFirst("255", "")))
    intercept[IllegalArgumentException] { DigitRecognizerPipeline.read(spark, missing, training = false) }
    val invalidLabel = writeCsv(dir.resolve("label.csv"), Seq("label," + header, "10," + image(0)))
    intercept[IllegalArgumentException] { DigitRecognizerPipeline.read(spark, invalidLabel, training = true) }
    val blank = writeCsv(dir.resolve("blank.csv"), Seq(header, image(0), "", image(1)))
    intercept[IllegalArgumentException] { DigitRecognizerPipeline.read(spark, blank, training = false) }
  }
}
