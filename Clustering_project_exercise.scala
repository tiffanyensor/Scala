/////////////////////////////////
// K MEANS PROJECT EXERCISE ////
///////////////////////////////

// Your task will be to try to cluster clients of a Wholesale Distributor
// based off of the sales of some product categories

// Source of the Data
//http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

// Here is the info on the data:
// 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
// 2)	MILK: annual spending (m.u.) on milk products (Continuous);
// 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
// 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
// 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
// 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
// 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
// 8)	REGION: customers Region- Lisnon, Oporto or Other (Nominal)

// Use K-means clustering with k = 3
// Experiment with other k values and use the blow method to determine the best one

////////////////////////////////////
//          SOLUTION         //////
//////////////////////////////////

// Set the Error reporting level
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Import Libraries and Functions
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Start a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Load in the data and extract the features
val dataset = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load("Wholesale customers data.csv")

val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

// Transform the data into an array of features
val assembler = (new VectorAssembler()
                .setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"))
                .setOutputCol("features"))

val training_data = assembler.transform(feature_data).select($"features")

// Apply K-means algorithm with k = 3
val kmeans = new KMeans().setK(3)
val model = kmeans.fit(training_data)

// Calculate the Within Set Sum of Squared Error to evaluate the model
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")


//////////////////////////////////////////
//  RESULTS FOR DIFFERENT K VALUES  /////
////////////////////////////////////////


// k = 2
// Within Set Sum of Squared Errors = 1.1322217166917542E11

// k = 3
// Within Set Sum of Squared Errors = 8.033326561848463E10

// k = 4
// Within Set Sum of Squared Errors = 6.751133343524618E10

// k = 5
// Within Set Sum of Squared Errors = 5.347110507965461E10

// k = 6
// Within Set Sum of Squared Errors = 5.074394121648486E10

// k = 7
// Within Set Sum of Squared Errors = 4.53150413353831E10

// k = 8
// Within Set Sum of Squared Errors = 3.803219660335069E10

// k = 10
// Within Set Sum of Squared Errors = 3.4944495644563416E10

// k = 15
// Within Set Sum of Squared Errors = 2.2610134721878826E10

// k = 20
// Within Set Sum of Squared Errors = 1.7737429008474327E10
