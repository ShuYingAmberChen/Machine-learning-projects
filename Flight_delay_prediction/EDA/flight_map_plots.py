# Databricks notebook source
!pip install geopandas
!pip install plotly

# COMMAND ----------

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quarter data set

# COMMAND ----------

#Reading in dataset from team mount
df = spark.read.parquet("/mnt/team11/all_flight_weather_3m")
display(df)

# COMMAND ----------

#Taking only the columns relevant to geographic visualization from the full dataset
df2 = df[["ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE","ORIGIN_CITY_NAME","DEST_CITY_NAME"]]

# COMMAND ----------

#Grouping the data to identify how many flights traveled from each Origin-Destination combination
counts = df2.groupby(["ORIG_LATITUDE","ORIG_LONGITUDE","DEST_LATITUDE","DEST_LONGITUDE"]).agg({"ORIG_LATITUDE":'count',                                                                                       "ORIGIN_CITY_NAME":'min',"DEST_CITY_NAME":'min'})

display(counts)

# COMMAND ----------

#Converting the Spark Dataframe into a Pandas Dataframe. This step is necessary to use NetworkX.
counts_df = counts.toPandas()

# COMMAND ----------

#First attempt at plotting - Includes both Atlanta and Chicago data
fig = go.Figure()

#Zip together longitudes and latitudes in order which form the nodes to connect 
source_to_dest = zip(counts_df["ORIG_LATITUDE"], counts_df["DEST_LATITUDE"],counts_df["ORIG_LONGITUDE"],counts_df["DEST_LONGITUDE"],counts_df["count"])

## Loop thorugh each flight entry, use linewidth to show how common a particular trip is
for slat,dlat, slon, dlon, num_flights in source_to_dest:
    fig.add_trace(go.Scattergeo( 
      lat = [slat,dlat], 
      lon = [slon, dlon], 
      mode = 'lines', 
      line = dict(width = num_flights/100, color="red")))

#Titling the plot and making some formatting changes
fig.update_layout(title_text = 'Connection Map Depicting Flights in First Quarter Dataset',height=700, width=900, margin={"t":0,"b":0,"l":0, "r":0, "pad":0}, showlegend=False)

fig.show()


# COMMAND ----------

#Splitting up the flights from Chicago and the flights from Atlanta
counts_df_chicago = counts_df[counts_df['ORIG_LATITUDE'] == 41.995]
counts_df_atlanta = counts_df[counts_df['ORIG_LATITUDE'] == 33.6301]

# COMMAND ----------

#Plotting Chicago flights
fig = go.Figure()

source_to_dest = zip(counts_df_chicago["ORIG_LATITUDE"], counts_df_chicago["DEST_LATITUDE"], counts_df_chicago["ORIG_LONGITUDE"], counts_df_chicago["DEST_LONGITUDE"], counts_df_chicago["count(ORIG_LATITUDE)"])

for slat,dlat, slon, dlon, num_flights in source_to_dest:
    fig.add_trace(go.Scattergeo(
      lat = [slat,dlat], 
      lon = [slon, dlon], 
      mode = 'lines', 
      line = dict(width = num_flights/300, color="darkblue", dash="solid")))

# Logic to create labels of source and destination cities of flights
cities = counts_df_chicago["min(ORIGIN_CITY_NAME)"].values.tolist()+counts_df_chicago["min(DEST_CITY_NAME)"].values.tolist()
    
fig.add_trace(
    go.Scattergeo(
                lon = counts_df_chicago["ORIG_LONGITUDE"].values.tolist()+counts_df_chicago["DEST_LONGITUDE"].values.tolist(),
                lat = counts_df_chicago["ORIG_LATITUDE"].values.tolist()+counts_df_chicago["DEST_LATITUDE"].values.tolist(),
                hoverinfo = 'text',
                text = cities,
                mode = 'markers',
                marker = dict(color = 'black', opacity=0.5, size = 10))
)

#Changing the colors and scope of the map
fig.update_geos(
    visible=False, resolution=50, scope="usa",
    showcountries=True, countrycolor="Black",
    showsubunits=True, subunitcolor="grey"
)

fig.update_layout(title_text = "Connection Map Depicting Flights from O'Hare in First Quarter Dataset",height=700, width=900, margin={"t":0,"b":0,"l":0, "r":0, "pad":0}, showlegend=False)

fig.show()


# COMMAND ----------

#Plotting flights from Atlanta
fig = go.Figure()

source_to_dest = zip(counts_df_atlanta["ORIG_LATITUDE"], counts_df_atlanta["DEST_LATITUDE"], counts_df_atlanta["ORIG_LONGITUDE"], counts_df_atlanta["DEST_LONGITUDE"], counts_df_atlanta["count(ORIG_LATITUDE)"])

for slat,dlat, slon, dlon, num_flights in source_to_dest:
    fig.add_trace(go.Scattergeo(
      lat = [slat,dlat], 
      lon = [slon, dlon], 
      mode = 'lines', 
      line = dict(width = num_flights/300, color="darkblue", dash="solid")))

cities = counts_df_atlanta["min(ORIGIN_CITY_NAME)"].values.tolist()+counts_df_atlanta["min(DEST_CITY_NAME)"].values.tolist()
    
fig.add_trace(
    go.Scattergeo(
                lon = counts_df_atlanta["ORIG_LONGITUDE"].values.tolist()+counts_df_atlanta["DEST_LONGITUDE"].values.tolist(),
                lat = counts_df_atlanta["ORIG_LATITUDE"].values.tolist()+counts_df_atlanta["DEST_LATITUDE"].values.tolist(),
                hoverinfo = 'text',
                text = cities,
                mode = 'markers',
                marker = dict(color = 'black', opacity=0.5, size = 10))
)

fig.update_geos(
    visible=False, resolution=50, scope="usa",
    showcountries=True, countrycolor="Black",
    showsubunits=True, subunitcolor="grey"
)

fig.update_layout(title_text = "Connection Map Depicting Flights from ATL in First Quarter Dataset",height=700, width=900, margin={"t":0,"b":0,"l":0, "r":0, "pad":0}, showlegend=False)

fig.show()


# COMMAND ----------


