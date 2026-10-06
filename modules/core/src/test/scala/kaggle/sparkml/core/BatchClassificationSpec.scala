package kaggle.sparkml.core

import java.nio.file.Files

class BatchClassificationSpec extends SparkSuite {
  test("CLI rejects invalid holdout fractions, missing values and unknown tuning options") {
    Seq("0", "1", "NaN", "Infinity", "-0.1").foreach { value =>
      intercept[IllegalArgumentException] {
        BatchClassification.parse(List("train", "--validation-fraction", value), "model", Set.empty)
      }
    }
    intercept[IllegalArgumentException] {
      BatchClassification.parse(List("predict", "--model"), "model", Set.empty)
    }
    intercept[IllegalArgumentException] {
      BatchClassification.parse(List("--unknown", "10"), "model", Set.empty)
    }
    val config = BatchClassification.parse(List("predict", "--input", "unlabeled.csv"), "saved", Set.empty)
    assert(config.mode == "predict" && config.inputPath == "unlabeled.csv" && config.modelPath == "saved")
  }

  test("single-file CSV writer preserves an existing submission unless overwrite is explicit") {
    val session = spark
    import session.implicits._
    val output = Files.createTempDirectory("batch-csv-").resolve("submission.csv")
    writeCsv(output, Seq("Id,Label", "1,old"))
    val predictions = Seq((1, "new"), (2, "new")).toDF("Id", "Label")
    intercept[IllegalArgumentException] {
      BatchClassification.writeSingleCsv(predictions, output.toString, overwrite = false)
    }
    assert(readLines(output).last == "1,old")
    BatchClassification.writeSingleCsv(predictions, output.toString, overwrite = true)
    assert(readLines(output) == Vector("Id,Label", "1,new", "2,new"))
    intercept[IllegalArgumentException] {
      BatchClassification.writeSingleCsv(predictions, output.getParent.toString, overwrite = true)
    }
  }
}
