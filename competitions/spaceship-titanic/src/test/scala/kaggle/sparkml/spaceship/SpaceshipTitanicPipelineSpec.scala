package kaggle.sparkml.spaceship

import java.nio.file.Files

import kaggle.sparkml.core.{BatchClassification, BatchConfig, SparkSuite}
import org.apache.spark.ml.PipelineModel

class SpaceshipTitanicPipelineSpec extends SparkSuite {
  private val header = "PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name"

  private def passenger(id: String, positive: Boolean, missing: Boolean = false): String = {
    val spend = if (positive) "0" else "100"
    val age = if (missing) "" else "25"
    val planet = if (missing) "UnknownPlanet" else "Earth"
    s"$id,$planet,${if (positive) "True" else "False"},F/12/S,TRAPPIST-1e,$age,False,$spend,$spend,$spend,$spend,$spend,Test Passenger"
  }

  test("train saves preprocessing and predict loads it without accessing training data") {
    val dir = Files.createTempDirectory("spaceship-pipeline-")
    val training = writeCsv(dir.resolve("train.csv"), Seq(header + ",Transported") ++ (1 to 100).map { i =>
      passenger(f"$i%04d" + "_01", i % 2 == 0) + s",${if (i % 2 == 0) "True" else "False"}"
    })
    val input = writeCsv(dir.resolve("test.csv"), Seq(header,
      passenger("0002_02", positive = false), passenger("0001_02", positive = true, missing = true)))
    val modelPath = dir.resolve("model").toString
    intercept[IllegalArgumentException] {
      BatchClassification.execute(spark, BatchConfig(mode = "train", trainPath = training,
        modelPath = dir.toString, overwrite = true), SpaceshipTitanicPipeline)
    }
    BatchClassification.execute(spark, BatchConfig(mode = "train", trainPath = training,
      modelPath = modelPath, tuning = Map("--num-trees" -> "5", "--max-depth" -> "3")), SpaceshipTitanicPipeline)
    val inputFrame = SpaceshipTitanicPipeline.read(spark, input, training = false)
    assert(!inputFrame.columns.contains("label"))
    val loaded = PipelineModel.load(modelPath)
    val expected = SpaceshipTitanicPipeline.submission(loaded.transform(inputFrame)).collect()
      .map(row => s"${row.getString(0)},${row.getString(1)}").toVector
    assert(expected == Vector("0001_02,True", "0002_02,False"))
    val output = dir.resolve("submission.csv")
    BatchClassification.execute(spark, BatchConfig(mode = "predict", trainPath = "does-not-exist.csv",
      inputPath = input, outputPath = output.toString, modelPath = modelPath), SpaceshipTitanicPipeline)
    val lines = readLines(output)
    assert(lines == Vector("PassengerId,Transported") ++ expected)
    assert(lines(1).startsWith("0001_02,"))
    assert(lines.tail.forall(line => line.endsWith(",True") || line.endsWith(",False")))
    assert(lines.length == 3)
  }

  test("duplicate identifiers, missing columns and invalid target values are rejected") {
    val dir = Files.createTempDirectory("spaceship-invalid-")
    val duplicate = writeCsv(dir.resolve("duplicate.csv"), Seq(header,
      passenger("0001_01", positive = true), passenger("0001_01", positive = false)))
    intercept[IllegalArgumentException] { SpaceshipTitanicPipeline.read(spark, duplicate, training = false) }
    val missing = writeCsv(dir.resolve("missing.csv"), Seq("PassengerId", "0001_01"))
    intercept[IllegalArgumentException] { SpaceshipTitanicPipeline.read(spark, missing, training = false) }
    val invalid = writeCsv(dir.resolve("invalid.csv"), Seq(header + ",Transported",
      passenger("0001_01", positive = true) + ",maybe"))
    intercept[IllegalArgumentException] { SpaceshipTitanicPipeline.read(spark, invalid, training = true) }
  }
}
