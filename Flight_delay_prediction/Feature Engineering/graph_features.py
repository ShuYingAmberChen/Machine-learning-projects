# Databricks notebook source
!pip install networkx

# COMMAND ----------

import networkx as nx
import pandas as pd
from pyspark.sql import SparkSession

# COMMAND ----------

#Reading in first quarter of data
df = spark.read.parquet("/mnt/team11/all_flight_weather_3m").toPandas()
df.head()

# COMMAND ----------

#Taking only relevant data from quarter data. 
df2 = df[["ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE","ORIGIN_CITY_NAME","DEST_CITY_NAME"]]

#Aggregating in order to identify the number of flights from each source/destination
counts = df2.groupby(["ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE"]).agg({"ORIG_LATITUDE":'count',                                                                                       "ORIGIN_CITY_NAME":'min',"DEST_CITY_NAME":'min'})

counts.rename(columns={'ORIG_LATITUDE':'weight'},inplace=True)
counts.head()

# COMMAND ----------

#Creating a directed graph using NetworkX
g = nx.from_pandas_edgelist(df, source='ORIGIN_CITY_NAME', target='DEST_CITY_NAME', create_using=nx.DiGraph())

# COMMAND ----------

#Adding the weight column to each origin/destination in the graph
for index, row in counts.iterrows():
  g[row.ORIGIN_CITY_NAME][row.DEST_CITY_NAME]['weight'] = row.weight

# COMMAND ----------

#Calculating degree centrality using NetworkX. 
#Also changing the index to be numbers instead of each city_name.
degree_centrality = nx.degree_centrality(g)
degree_centrality_df = pd.DataFrame.from_dict(centrality, orient='index').reset_index()
degree_centrality_df.rename(columns={'index':'city_name',0:'degree_centrality'}, inplace=True)

degree_centrality_df.sort_values(by='degree_centrality', ascending=False)

# COMMAND ----------

#Calculating pageRank using NetworkX. 
#Also changing the index to be numbers instead of each city_name.
pagerank = nx.pagerank(g)
pagerank_df = pd.DataFrame.from_dict(pagerank, orient='index').reset_index()
pagerank_df.rename(columns={'index':'city_name',0:'page_rank'}, inplace=True)

pagerank_df.sort_values(by='page_rank', ascending=False)

# COMMAND ----------

#Merging together PageRank and Degree Centrality dataframes on city_name
merged = pd.merge(degree_centrality_df, pagerank_df, on='city_name').sort_values(by='degree_centrality', ascending=False)

# COMMAND ----------

#Converting to a Spark DataFrame so that I can push to the team repo
sparkDF=spark.createDataFrame(merged) 
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

#Pushing to team repo
sparkDF.write.parquet("/mnt/team11/graph_metrics")

# COMMAND ----------

#Reading from team repo
display(spark.read.parquet("/mnt/team11/graph_metrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC #5M Graph Features - Same process as 3M

# COMMAND ----------

#Reading in total graph data
df_5y = spark.read.parquet("/mnt/team11/all_flight_weather_5y/*")

# COMMAND ----------

df_graph_5y = df_5y.select("ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE","ORIGIN_CITY_NAME","DEST_CITY_NAME").toPandas()

# COMMAND ----------

counts = df_graph_5y.groupby(["ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE"]).agg({"ORIG_LATITUDE":'count',                                                                                       "ORIGIN_CITY_NAME":'min',"DEST_CITY_NAME":'min'})

counts.rename(columns={'ORIG_LATITUDE':'weight'},inplace=True)
counts.head()

# COMMAND ----------

g = nx.from_pandas_edgelist(df_graph_5y, source='ORIGIN_CITY_NAME', target='DEST_CITY_NAME', create_using=nx.DiGraph())

# COMMAND ----------

counts = df_graph_5y.groupby(["ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE"]).agg({"ORIG_LATITUDE":'count',                                                                                       "ORIGIN_CITY_NAME":'min',"DEST_CITY_NAME":'min'})

counts.rename(columns={'ORIG_LATITUDE':'weight'},inplace=True)
counts.head()

g = nx.from_pandas_edgelist(df_graph_5y, source='ORIGIN_CITY_NAME', target='DEST_CITY_NAME', create_using=nx.DiGraph())

for index, row in counts.iterrows():
  g[row.ORIGIN_CITY_NAME][row.DEST_CITY_NAME]['weight'] = row.weight

# COMMAND ----------

pagerank = nx.pagerank(g)
pagerank_df = pd.DataFrame.from_dict(pagerank, orient='index').reset_index()
pagerank_df.rename(columns={'index':'city_name',0:'page_rank'}, inplace=True)

pagerank_df.sort_values(by='page_rank', ascending=False)

# COMMAND ----------

degree_centrality = nx.degree_centrality(g)
degree_centrality_df = pd.DataFrame.from_dict(degree_centrality, orient='index').reset_index()
degree_centrality_df.rename(columns={'index':'city_name',0:'degree_centrality'}, inplace=True)

degree_centrality_df.sort_values(by='degree_centrality', ascending=False)

# COMMAND ----------

merged = pd.merge(degree_centrality_df, pagerank_df, on='city_name').sort_values(by='degree_centrality', ascending=False)

# COMMAND ----------

display(merged)

# COMMAND ----------

sparkDF=spark.createDataFrame(merged) 
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

sparkDF.write.parquet("/mnt/team11/graph_metrics_5M")

# COMMAND ----------

display(spark.read.parquet("/mnt/team11/graph_metrics_5M")).

# COMMAND ----------


