package kaggle.sparkml.spaceship

import java.nio.file.Files

import kaggle.sparkml.core.{BatchClassification, BatchConfig, SparkSuite}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.feature.SQLTransformer

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

  test("GBT pipeline retains extended features and settings after loading") {
    val dir = Files.createTempDirectory("spaceship-gbt-")
    val training = writeCsv(dir.resolve("train.csv"), Seq(header + ",Transported") ++ (1 to 60).map { i =>
      passenger(f"$i%04d" + "_01", i % 2 == 0) + s",${if (i % 2 == 0) "True" else "False"}"
    })
    val frame = SpaceshipTitanicPipeline.read(spark, training, training = true)
    val config = BatchConfig(tuning = Map("--classifier" -> "gbt", "--max-iter" -> "3",
      "--max-depth" -> "2", "--step-size" -> "0.1", "--feature-set" -> "extended"))
    val fitted = SpaceshipTitanicPipeline.pipeline(config).fit(frame)
    val path = dir.resolve("model").toString
    fitted.write.save(path)
    Files.delete(java.nio.file.Paths.get(training))
    val loaded = PipelineModel.load(path)
    val classifier = loaded.stages.last.asInstanceOf[GBTClassificationModel]
    assert(classifier.getMaxIter == 3)
    assert(classifier.getMaxDepth == 2)
    assert(classifier.getStepSize == 0.1)
    val input = writeCsv(dir.resolve("test.csv"), Seq(header,
      passenger("0001_02", positive = true, missing = true), passenger("0002_02", positive = false)))
    val inputFrame = SpaceshipTitanicPipeline.read(spark, input, training = false)
    val transformed = loaded.transform(inputFrame)
    assert(transformed.select("GroupNumber", "MissingSpendCount", "SpendServiceCount", "SpaShare")
      .collect().toVector == fitted.transform(inputFrame)
      .select("GroupNumber", "MissingSpendCount", "SpendServiceCount", "SpaShare").collect().toVector)
    val result = SpaceshipTitanicPipeline.submission(transformed).collect()
    assert(result.map(row => s"${row.getString(0)},${row.getString(1)}").toVector ==
      Vector("0001_02,True", "0002_02,False"))
  }

  test("extended spending features distinguish missing values from zeros and avoid division by zero") {
    val dir = Files.createTempDirectory("spaceship-features-")
    val input = writeCsv(dir.resolve("test.csv"), Seq(header,
      "0123_01,Earth,False,F/12/S,TRAPPIST-1e,25,False,100,,0,300,0,Test Passenger",
      "0123_02,Earth,True,F/12/S,TRAPPIST-1e,25,False,0,0,0,0,0,Test Passenger",
      "0124_01,Earth,,F/12/S,TRAPPIST-1e,25,False,,,,,,Test Passenger"))
    val frame = SpaceshipTitanicPipeline.read(spark, input, training = false)
    val basic = SpaceshipTitanicPipeline.pipeline(BatchConfig()).getStages.head.asInstanceOf[SQLTransformer]
      .transform(frame)
    assert(!basic.columns.contains("GroupNumber"))
    val extended = SpaceshipTitanicPipeline.pipeline(BatchConfig(tuning = Map("--feature-set" -> "extended")))
      .getStages.head.asInstanceOf[SQLTransformer].transform(frame)
    val rows = extended.selectExpr("PassengerId", "GroupNumber", "CAST(MissingSpendCount AS DOUBLE)",
      "CAST(SpendServiceCount AS DOUBLE)", "RoomServiceShare", "SpaShare", "FoodCourtShare")
      .orderBy("PassengerId").collect()
    assert(rows(0).getDouble(1) == 123.0)
    assert(rows(0).getDouble(2) == 1.0)
    assert(rows(0).getDouble(3) == 2.0)
    assert(rows(0).getDouble(4) == 0.25)
    assert(rows(0).getDouble(5) == 0.75)
    assert(rows(0).getDouble(6) == 0.0)
    assert(rows(1).getDouble(2) == 0.0)
    assert(rows(2).getDouble(2) == 5.0)
    rows.tail.foreach { row =>
      assert(row.getDouble(3) == 0.0)
      (4 to 6).foreach(index => assert(row.getDouble(index) == 0.0))
    }
  }

  test("invalid classifier and GBT parameters are rejected before fitting") {
    Seq("--classifier" -> "unknown", "--feature-set" -> "unknown", "--max-iter" -> "0", "--step-size" -> "0",
      "--step-size" -> "1.1", "--step-size" -> "NaN").foreach { option =>
      intercept[IllegalArgumentException] {
        SpaceshipTitanicPipeline.pipeline(BatchConfig(tuning = Map(option)))
      }
    }
  }
}
