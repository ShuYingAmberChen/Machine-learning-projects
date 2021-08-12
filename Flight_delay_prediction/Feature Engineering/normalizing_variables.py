# Databricks notebook source
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import stddev, mean, col

# COMMAND ----------

#REading in both datasets
df_spark_3m = spark.read.parquet("/mnt/team11/all_flight_weather_3m")
df_spark_5y = spark.read.parquet("/mnt/team11/all_flight_weather_5y/*")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Initial Model Features
# MAGIC Airline: Carrier, departure airport, arrival airport, distance, scheduled departure time, scheduled arrival time, time of day, day of week
# MAGIC 
# MAGIC Weather: Elevation (ORIG_ELEVATION, DEST_ELEVATION), wind direction (ORIG_direction_angle, DEST_direction_angle), wind speed (ORIG_speed, DEST_speed), ceiling height (ORIG_ceiling_height, DEST_ceiling_height), visibility distance (ORIG_vis_distance, DEST_vis_distance), air temperature (ORIG_air_temp, DEST_air_temp), dew point temperature (ORIG_dew_point, DEST_dew_point), sea level pressure (ORIG_sea_level_pressure, DEST_sea_level_pressure), liquid precipitation (ORIG_precipitation, DEST_precipitation)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3M Dataset

# COMMAND ----------

#Numerical data from the weather dataset
num_var = ["ORIG_ELEVATION", "DEST_ELEVATION", "ORIG_direction_angle", "DEST_direction_angle","ORIG_speed", "DEST_speed", "ORIG_ceiling_height", "DEST_ceiling_height", "ORIG_vis_distance", "DEST_vis_distance", "ORIG_air_temp", "ORIG_dew_point", "DEST_dew_point", "ORIG_sea_level_pressure", "DEST_sea_level_pressure", "CRS_DEP_TIME", "CRS_ARR_TIME", "ORIG_precipitation_hrs", "ORIG_precipitation_depth", "DEST_precipitation_hrs", "DEST_precipitation_depth"]

#Creating an index column for the total dataset, then only keeping the the index
df_joined=df_spark_3m.select("*").withColumn("id", monotonically_increasing_id()).select("id")

#For each numerical variable, first create a mean and standard deviation column. Cross Join this with the large dataset.
#Compute the normalized variable by subtracting the mean and dividing by standard deviation.
#Add on a column for the index
#Join this with df_joined, which ends up just holding the normalized variables
for i in num_var:
  df_temp = df_spark_3m \
    .select(mean(i) \
    .alias("mean_"+i), stddev(i).alias("std_"+i)) \
    .crossJoin(df_spark_3m) \
    .withColumn(i+"_scaled", (col(i)-col("mean_"+i))/col("std_"+i)) \
    .select(i+"_scaled") \
    .withColumn("id", monotonically_increasing_id())
  df_joined = df_temp.join(df_joined, on = "id", how='inner')

# COMMAND ----------

#Check that the normalized dataset has as many rows as the original
df_joined.count() == df_spark_3m.count()

# COMMAND ----------

#Write to team mount
df_joined.write.parquet("/mnt/team11/normalized_variables_3m")

# COMMAND ----------

#Test reading from the mount
spark.read.parquet("/mnt/team11/normalized_variables_3m").count()

# COMMAND ----------

# MAGIC %md
# MAGIC #Total Dataset - Same process as 3M

# COMMAND ----------

num_var = ["ORIG_ELEVATION", "DEST_ELEVATION", "ORIG_direction_angle", "DEST_direction_angle","ORIG_speed", "DEST_speed", "ORIG_ceiling_height", "DEST_ceiling_height", "ORIG_vis_distance", "DEST_vis_distance", "ORIG_air_temp", "ORIG_dew_point", "DEST_dew_point", "ORIG_sea_level_pressure", "DEST_sea_level_pressure", "CRS_DEP_TIME", "CRS_ARR_TIME", "ORIG_precipitation_hrs", "ORIG_precipitation_depth", "DEST_precipitation_hrs", "DEST_precipitation_depth"]

df_joined_5y = df_spark_5y.select("*").withColumn("id", monotonically_increasing_id()).select("id")

for i in num_var:
  df_temp_5y = df_spark_5y \
    .select(mean(i) \
    .alias("mean_"+i), stddev(i).alias("std_"+i)) \
    .crossJoin(df_spark_5y) \
    .withColumn(i+"_scaled", (col(i)-col("mean_"+i))/col("std_"+i)) \
    .select(i+"_scaled") \
    .withColumn("id", monotonically_increasing_id())
  df_joined_5y = df_temp_5y.join(df_joined_5y, on = "id", how='inner')

# COMMAND ----------

df_joined_5y.count() == df_spark_5y.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Pushing to Team Storage

# COMMAND ----------

df_joined_5y.write.parquet("/mnt/team11/normalized_variables_v2")

# COMMAND ----------

a = spark.read.parquet("/mnt/team11/normalized_variables_v2")

# COMMAND ----------

display(a)

# COMMAND ----------

a.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Normalizing Graph Features - Same process as previous

# COMMAND ----------

graph = spark.read.parquet("/mnt/team11/graph_metrics")

# COMMAND ----------

num_var = ["degree_centrality", "page_rank"]

graph_joiner = graph.select("city_name")

for i in num_var:
  df_temp_graph = graph \
    .select(mean(i) \
    .alias("mean_"+i), stddev(i).alias("std_"+i)) \
    .crossJoin(graph) \
    .withColumn(i+"_scaled", (col(i)-col("mean_"+i))/col("std_"+i)) \
    .select( "city_name", i+"_scaled")
  graph_joiner = df_temp_graph.join(graph_joiner, on = "city_name", how='inner')

# COMMAND ----------

display(graph_joiner)

# COMMAND ----------

graph_joiner.write.parquet("/mnt/team11/normalized_graph")

# COMMAND ----------

display(spark.read.parquet("/mnt/team11/normalized_graph"))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Graph 5M - Same process as previous

# COMMAND ----------

graph2 = spark.read.parquet("/mnt/team11/graph_metrics_5M")

# COMMAND ----------

num_var = ["degree_centrality", "page_rank"]

graph_joiner = graph2.select("city_name")

for i in num_var:
  df_temp_graph = graph2 \
    .select(mean(i) \
    .alias("mean_"+i), stddev(i).alias("std_"+i)) \
    .crossJoin(graph2) \
    .withColumn(i+"_scaled", (col(i)-col("mean_"+i))/col("std_"+i)) \
    .select( "city_name", i+"_scaled")
  graph_joiner = df_temp_graph.join(graph_joiner, on = "city_name", how='inner')

# COMMAND ----------

display(graph_joiner)

# COMMAND ----------

graph_joiner.write.parquet("/mnt/team11/normalized_graph_5M")

# COMMAND ----------

display(spark.read.parquet("/mnt/team11/normalized_graph_5M"))

# COMMAND ----------


