ThisBuild / organization := "dev.hardbox"
ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version := "0.1.0-SNAPSHOT"

val sparkVersion = "3.5.9"

lazy val commonSettings = Seq(
  resolvers += Resolver.mavenCentral,
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    ("ml.dmlc" %% "xgboost4j-spark" % "3.2.0")
      .exclude("com.esotericsoftware", "kryo")
  ),
  Compile / run / fork := true,
  Test / fork := true,
  Compile / run / connectInput := true,
  Compile / run / javaOptions ++= Seq(
    "-Dspark.ui.enabled=false",
    "-Dlog4j2.configurationFile=log4j2.properties"
  )
)

lazy val root = (project in file("."))
  .aggregate(titanic)
  .settings(
    name := "spark-ml-kaggle",
    publish / skip := true
  )

lazy val titanic = (project in file("competitions/titanic"))
  .settings(commonSettings)
  .settings(
    name := "kaggle-titanic-spark",
    Compile / run / mainClass := Some("kaggle.sparkml.titanic.TitanicPipeline")
  )
