ThisBuild / organization := "dev.hardbox"
ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version := "0.1.0-SNAPSHOT"

val sparkVersion = "3.5.9"

lazy val commonSettings = Seq(
  resolvers += Resolver.mavenCentral,
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion
  ),
  Compile / run / fork := true,
  Test / fork := true,
  Compile / run / connectInput := true,
  Compile / run / javaOptions ++= Seq(
    "-Dspark.ui.enabled=false",
    "-Dlog4j2.configurationFile=log4j2.properties"
  )
)

lazy val xgboostSettings = Seq(
  libraryDependencies += ("ml.dmlc" %% "xgboost4j-spark" % "3.2.0")
    .exclude("com.esotericsoftware", "kryo")
)

lazy val batchSettings = Seq(
  libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test,
  Test / parallelExecution := false,
  Test / javaOptions ++= Seq("-Xmx2g", "-Dspark.ui.enabled=false"),
  Compile / run / javaOptions += "-Xmx2g"
)

lazy val root = (project in file("."))
  .aggregate(titanic, housePrices, core, spaceshipTitanic, digitRecognizer)
  .settings(
    name := "spark-ml-kaggle",
    publish / skip := true
  )

lazy val titanic = (project in file("competitions/titanic"))
  .settings(commonSettings)
  .settings(xgboostSettings)
  .settings(
    name := "kaggle-titanic-spark",
    Compile / run / mainClass := Some("kaggle.sparkml.titanic.TitanicPipeline")
  )

lazy val housePrices = (project in file("competitions/house-prices"))
  .settings(commonSettings)
  .settings(xgboostSettings)
  .settings(
    name := "kaggle-house-prices-spark",
    Compile / run / mainClass := Some("kaggle.sparkml.houseprices.HousePricesPipeline")
  )

lazy val core = (project in file("modules/core"))
  .settings(commonSettings)
  .settings(batchSettings)
  .settings(name := "kaggle-batch-core")

lazy val spaceshipTitanic = (project in file("competitions/spaceship-titanic"))
  .dependsOn(core % "compile->compile;test->test")
  .settings(commonSettings)
  .settings(batchSettings)
  .settings(
    name := "kaggle-spaceship-titanic-spark",
    Compile / run / mainClass := Some("kaggle.sparkml.spaceship.SpaceshipTitanicPipeline")
  )

lazy val digitRecognizer = (project in file("competitions/digit-recognizer"))
  .dependsOn(core % "compile->compile;test->test")
  .settings(commonSettings)
  .settings(batchSettings)
  .settings(
    name := "kaggle-digit-recognizer-spark",
    Compile / run / mainClass := Some("kaggle.sparkml.digits.DigitRecognizerPipeline")
  )
