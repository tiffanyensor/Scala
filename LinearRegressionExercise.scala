// Linear Regression Exercise using Scala
// Estimate "Yearly Amount Spent"

// Import Libraries and functions
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Start Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Read in the data from the "Clean-Ecommerce.csv" file
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
data.printSchema()

// Create a dataframe to hold the data for Linear regression
// First column for the label (Yearly Amount Spent)
// Second column for features (all other numerical columns)
val df = (data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership"))
df.printSchema()

// Use VectorAssembler to create the array of features
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")
val output = assembler.transform(df).select($"label", $"features")
output.show()

// Create a Linear Regression Model object and fit the data
val lr = new LinearRegression()
val lrModel = lr.fit(output)

// Print the coefficients, intercept, and R^2 value
val trainingSummary = lrModel.summary

println("COEFFICIENTS", lrModel.coefficients)
println("INTERCEPT", lrModel.intercept)
println("R-SQUARED", trainingSummary.r2)
