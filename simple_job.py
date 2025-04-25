from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("SimpleExample") \
    .getOrCreate()

# Example DataFrame
data = [("Aadarsh", 25), ("Ram", 30), ("Shyam", 28)]
df = spark.createDataFrame(data, ["name", "age"])

# Show the data
df.show()

# Stop Spark session
spark.stop()
