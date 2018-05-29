//////////////////////////////////////////////
// LOGISTIC REGRESSION PROJECT //////////////
////////////////////////////////////////////

//  In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
//  This data set contains the following features:
//    'Daily Time Spent on Site': consumer time on site in minutes
//    'Age': cutomer age in years
//    'Area Income': Avg. Income of geographical area of consumer
//    'Daily Internet Usage': Avg. minutes a day consumer is on the internet
//    'Ad Topic Line': Headline of the advertisement
//    'City': City of consumer
//    'Male': Whether or not consumer was male
//    'Country': Country of consumer
//    'Timestamp': Time at which consumer clicked on Ad or closed window
//    'Clicked on Ad': 0 or 1 indicated clicking on Ad

// Import libraries and functions
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Set error reporting level
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create spark session and read in data
val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("advertising.csv")
data.printSchema()

// Print a sample row of data
val column_names = data.columns
val first_row = data.head(1)(0)

println("-----------------")
println("Example data row")
println("-----------------")

for (i <- Range(1,column_names.length)){
  println(column_names(i),first_row(i))
}

// Add a column for "hour" (extracted from timestamp)
val data_w_hour = data.withColumn("Hour", hour(data("Timestamp")))

// Select the DVs for the logistic regression and rename IV as "label"
val df = (data_w_hour.select(data("Clicked on Ad") as ("label"),
                  $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
                  )

// Create a "features" array containing the DVs
val assembler = (new VectorAssembler()
                .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Hour", "Male"))
                .setOutputCol("features")
                )

// Split the data into train set and test set
val Array(train_set, test_set) = df.randomSplit(Array(0.7,0.3), seed = 12345)

// Create logistic regression object
val lr = new LogisticRegression()

// Set up pipeline and fit the model
val pipeline = new Pipeline().setStages(Array(assembler, lr))
val model = pipeline.fit(train_set)
val results = model.transform(test_set)

// Evaluate model with confusion matrix
val predictions_and_labels = results.select($"prediction", $"label").as[(Double,Double)].rdd
val metrics = new MulticlassMetrics(predictions_and_labels)
println("-------------------")
println("Confusion matrix: ")
println("-------------------")
println(metrics.confusionMatrix)
