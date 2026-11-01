package kaggle.sparkml.disasters

import java.nio.file.Files

import kaggle.sparkml.core.{BatchClassification, BatchConfig, SparkSuite}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.{Vector => MLVector}

class DisasterTweetsPipelineSpec extends SparkSuite {
  private val header = "id,keyword,location,text"

  test("saved text pipeline predicts without training CSV and preserves tweet identifiers") {
    val dir = Files.createTempDirectory("disaster-tweets-")
    val training = writeCsv(dir.resolve("train.csv"), Seq(header + ",target") ++ (1 to 80).map { i =>
      if (i % 2 == 0) s"$i,flood,London,Emergency flood rescue evacuation,1"
      else s"$i,holiday,Paris,Happy holiday music picnic friends,0"
    })
    val input = writeCsv(dir.resolve("test.csv"), Seq(header,
      "30,new%20keyword,,\"Emergency flood, rescue\n evacuation \"\"now\"\" https://example.org @helper\"",
      "2,,,Happy holiday music picnic friends", "11,,,"))
    val model = dir.resolve("model").toString
    BatchClassification.execute(spark, BatchConfig(mode = "train", trainPath = training,
      modelPath = model, tuning = Map("--num-features" -> "256", "--max-iter" -> "10")), DisasterTweetsPipeline)
    Files.delete(dir.resolve("train.csv"))
    val frame = DisasterTweetsPipeline.read(spark, input, training = false)
    assert(!frame.columns.contains("label"))
    assert(frame.count() == 3)
    val loaded = PipelineModel.load(model)
    val words = loaded.transform(frame).filter("id = 30").select("words").head().getSeq[String](0)
    assert(words.contains("rescue") && words.contains("new") && words.contains("keyword"))
    assert(!words.exists(Set("https", "example", "org", "helper")))
    val predicted = DisasterTweetsPipeline.submission(loaded.transform(frame)).collect()
      .map(row => s"${row.getLong(0)},${row.getInt(1)}").toVector
    assert(predicted.head == "2,0")
    assert(predicted.last == "30,1")
    val output = dir.resolve("submission.csv")
    BatchClassification.execute(spark, BatchConfig(mode = "predict", trainPath = "missing.csv",
      inputPath = input, outputPath = output.toString, modelPath = model), DisasterTweetsPipeline)
    assert(readLines(output) == Vector("id,target") ++ predicted)
    intercept[IllegalArgumentException] {
      BatchClassification.execute(spark, BatchConfig(mode = "predict", inputPath = input,
        outputPath = output.toString, modelPath = model), DisasterTweetsPipeline)
    }
  }

  test("count vocabulary is fitted only on training tweets and stays frozen after loading") {
    val dir = Files.createTempDirectory("disaster-count-")
    val training = writeCsv(dir.resolve("train.csv"), Seq(header + ",target") ++ (1 to 60).map { i =>
      val rare = if (i == 1) " singularword" else ""
      if (i % 2 == 0) s"$i,,,Emergency flood rescue evacuation,1"
      else s"$i,,,Happy holiday music picnic friends$rare,0"
    })
    val fitted = DisasterTweetsPipeline.pipeline(BatchConfig(tuning = Map(
      "--vectorizer" -> "count", "--num-features" -> "256", "--max-iter" -> "10")))
      .fit(DisasterTweetsPipeline.read(spark, training, training = true))
    val model = dir.resolve("model").toString
    fitted.write.save(model)
    Files.delete(dir.resolve("train.csv"))
    val loaded = PipelineModel.load(model)
    val vocabulary = loaded.stages.collectFirst { case stage: CountVectorizerModel => stage.vocabulary }.get
    assert(vocabulary.contains("flood") && vocabulary.contains("holiday") && vocabulary.contains("flood rescue"))
    assert(!vocabulary.contains("singularword") && !vocabulary.contains("nebula"))
    assert(vocabulary.toVector == fitted.stages.collectFirst {
      case stage: CountVectorizerModel => stage.vocabulary.toVector
    }.get)
    val input = writeCsv(dir.resolve("test.csv"), Seq(header,
      "30,,,Emergency flood rescue", "2,,,Happy holiday music", "99,,,nebula", "100,,,"))
    val transformed = loaded.transform(DisasterTweetsPipeline.read(spark, input, training = false))
    transformed.filter("id >= 99").select("termCounts").collect().foreach { row =>
      val counts = row.getAs[MLVector](0)
      assert(counts.size == vocabulary.length && counts.numNonzeros == 0)
    }
    val output = dir.resolve("submission.csv")
    BatchClassification.execute(spark, BatchConfig(mode = "predict", trainPath = "missing.csv",
      inputPath = input, outputPath = output.toString, modelPath = model), DisasterTweetsPipeline)
    val lines = readLines(output)
    assert(lines.head == "id,target" && lines.length == 5)
    assert(lines(1) == "2,0" && lines(2) == "30,1")
    assert(lines.tail.map(_.split(",")(0)) == Vector("2", "30", "99", "100"))
    assert(lines.tail.forall(line => line.endsWith(",0") || line.endsWith(",1")))
  }

  test("invalid labels, duplicate or invalid IDs and missing columns are rejected") {
    val dir = Files.createTempDirectory("disaster-invalid-")
    Seq(
      Seq(header + ",target", "1,,,flood,2"),
      Seq(header + ",target", "1,,,flood,"),
      Seq(header + ",target", "1,,,flood,1", "01,,,music,0"),
      Seq(header + ",target", "-1,,,flood,1"),
      Seq(header + ",target", "9999999999999999999999999,,,flood,1"),
      Seq("id,text,target", "1,flood,1"),
      Seq(header + ",target")
    ).zipWithIndex.foreach { case (rows, index) =>
      val path = writeCsv(dir.resolve(s"invalid-$index.csv"), rows)
      intercept[IllegalArgumentException] { DisasterTweetsPipeline.read(spark, path, training = true) }
    }
  }

  test("F1 is computed for the positive class rather than a weighted average") {
    val session = spark
    import session.implicits._
    val predictions = Seq((1.0, 1.0), (1.0, 0.0), (0.0, 1.0),
      (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)).toDF("label", "prediction")
    assert(DisasterTweetsPipeline.validationMetrics(predictions).head._2 == 0.5)
    val noPositives = Seq((0.0, 0.0)).toDF("label", "prediction")
    assert(DisasterTweetsPipeline.validationMetrics(noPositives).head._2 == 0.0)
  }

  test("invalid tuning parameters are rejected") {
    Seq("--vectorizer" -> "unknown", "--num-features" -> "0", "--max-iter" -> "0", "--reg-param" -> "NaN",
      "--reg-param" -> "Infinity", "--reg-param" -> "-1", "--threshold" -> "1").foreach { option =>
      intercept[IllegalArgumentException] {
        DisasterTweetsPipeline.pipeline(BatchConfig(tuning = Map(option)))
      }
    }
  }
}
