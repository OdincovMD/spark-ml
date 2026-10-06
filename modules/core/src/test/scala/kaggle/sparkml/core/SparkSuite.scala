package kaggle.sparkml.core

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

abstract class SparkSuite extends AnyFunSuite with BeforeAndAfterAll {
  protected var spark: SparkSession = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder().master("local[2]").appName(getClass.getSimpleName)
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  override protected def afterAll(): Unit = {
    try if (spark != null) spark.stop()
    finally super.afterAll()
  }

  protected def writeCsv(path: Path, lines: Seq[String]): String = {
    Files.write(path, (lines.mkString("\n") + "\n").getBytes(StandardCharsets.UTF_8))
    path.toString
  }

  protected def readLines(path: Path): Vector[String] = {
    import scala.collection.JavaConverters._
    Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
  }
}
