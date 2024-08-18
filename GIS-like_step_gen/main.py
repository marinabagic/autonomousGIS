from graph_gen import generate_graph
from code_gen import generate_code, generate_assembly_code
import networkx as nx
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import matplotlib as plt


# model_id = "google/codegemma-7b"
# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16
# ).to("cuda:1")

# model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # vrlo dobar
# model_name = "HuggingFaceH4/zephyr-7b-beta"  # majstor halucinacija
# model_name = "teknium/OpenHermes-2.5-Mistral-7B"  # sjajan je
# model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"  # malo spor i malo losiji
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # ovaj model bi mogao biti svet, hvala, radi geopandas napokon ides
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # jedini koji radi u potpunosti
# model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"  # i ovaj radi lijepo
model_name = "bigcode/starcoder2-15b-instruct-v0.1"

pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto", token="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")

#1
# task = """
# Generate a graph (data structure) only, whose nodes are (1) a series of consecutive steps and (2) data to solve this question:  
# 1) Find out Census tracts that contain hazardous waste facilities, then compute and print out the population living in those tracts. The study area is North Carolina (NC), US.
# 2) Generate a population choropleth map for all tract polygons in NC, rendering the color by tract population; and then highlight the borders of tracts that have hazardous waste facilities. Please draw all polygons, not only the highlighted ones. The map size is 15*10 inches.
# """


#2
# task = r'''
# For each zipcode area in South Carolina (SC), calculate the distance from the centroid of the zipcode area to its nearest hospital, and then create a choropleth distance map of zipcode area polygons (unit: km), also show the hospital.
# '''

#3

task = r'''1) Draw a choropleth map to show the death rate (death/case) of COVID-19 among the countiguous US counties. Use the accumulated COVID-19 data of 2020.12.31 to compute the death rate. Use scheme ='quantiles' when plotting the map.  Set map projection to 'Conus Albers'. Set map size to 15*10 inches.  
2) Draw a scatter plot to show the correlation and trend line of the death rate with the senior resident rate, including the r-square and p-value. Set data point transparency to 50%, regression line as red. Set figure size to 15*10 inches.  
'''

#4

# task = r'''1) Show the monthly change rates of population mobility for each administrative regions in a France map. Each month is a sub-map in a map matrix. The base of the change rate is January 2020.
# 2) Draw a line chart to show the monthly change rate trends of all administrative regions. Th x-axis is month.'''
# data_locations_str = ["ESRI shapefile for France administrative regions: https://github.com/gladcolor/LLMGeo/raw/master/REST_API/France.zip. The 'GID_1' column is the administrative region code, 'NAME_1' column is the administrative region name.",
#             "REST API URL with parameters for mobility data access: http://gis.cas.sc.edu/GeoAnalytics/REST?operation=get_daily_movement_for_all_places&source=twitter&scale= world_first_level_admin&begin=01/01/2020&end=12/31/2020. The response is in CSV format. There are three columns in the response: place, date (format:2020-01-07), and intra_movement. 'place' column is the administrative region code, France administrative regions start with 'FRA'",
#             ]

# data_locations_str = ["NC hazardous waste facility ESRI shape file: https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/HW_Sites_EPSG4326.zip.",
#             "NC tract boundary shapefile: https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/tract_37_EPSG4326.zip. The tract ID column is 'GEOID', data types is integer.",
#             "NC tract population CSV file: https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/NC_tract_population.csv. The population is stored in 'TotalPopulation' column. The tract ID column is 'GEOID', data types is integer."
#             ]

# data_locations_str = [
# r"SC zipcode boundary shapefile: https://github.com/GIBDUSC/test/raw/master/sc_zip_boundary.zip, the map projection is WGS1984.",
# r"SC hospitals:  https://github.com/gladcolor/spatial_data/raw/master/South_Carolina/SC_hospitals_with_emergency_room_cleaned.csv, location columns: longitude in 'POINT_X' column, latitude in 'POINT_Y' column.",          
# ]

data_locations_str = [
                  r"COVID-19 data case in 2020 (county-level): https://github.com/nytimes/covid-19-data/raw/master/us-counties-2020.csv. This data is for daily accumulated COVID cases and deaths for each county in the US. There are 5 columns: date (format: 2021-02-01), county, state, fips, cases, deaths. ",   
                  r"Contiguous US county boundary (ESRI shapefile): https://github.com/gladcolor/spatial_data/raw/master/contiguous_counties.zip. The county FIPS column is 'GEOID'; map projection is EPSG:4269",
                  r"Census data (ACS2020): https://raw.githubusercontent.com/gladcolor/spatial_data/master/Demography/ACS2020_5year_county.csv. THe needed columns are: 'FIPS', 'Total Population', 'Total Population: 65 to 74 Years', 'Total Population: 75 to 84 Years', 'Total Population: 85 Years and Over'. Drop rows with NaN cells after loading the used columns.",
                 ]

# task = r''' Show the spatial distribution of the county level median income in the contigous US. Set figure size to (25,15)'''
# data_locations_str = ["You can use the Census API. Here is the API key: ae7be70727932dd6aed257692de3f344365d0678"]


# task = r'''
# 1) Show the 2020 human mobility monthly change rates of each administrative regions in a France choropleth map. Each month is a sub-map in a map matrix，12 months in total. All monthly maps need to use the same colorbar range (color scheme: coolwarm). The base of the change rate is January 2020. 
# 2) Draw a line chart to show the monthly change rate trends of all administrative regeions. Each region is a line (the region name is the legend), the x-axis is 2020 months.
# '''

# data_locations_str = ["ESRI shapefile for France administrative regions:" + \
#                   "https://github.com/gladcolor/LLM-Geo/raw/master/REST_API/France.zip. " + \
#                   "The 'GID_1' column is the administrative region code, 'NAME_1' column is the administrative region name.",
#                   "REST API url with parameters for daily human mobility data access:" + \
#                   "http://gis.cas.sc.edu/GeoAnalytics/REST?operation=get_daily_movement_for_all_places&source=twitter&scale=world_first_level_admin&begin=01/01/2020&end=12/31/2020." + \
#                   "The response is in CSV format. There are three columns in the response: " + \
#                   "place,date (format:2020-01-07), and intra_movement. 'place' column is the administractive region code of every country;" + \
#                   "codes for France administrative regions start with 'FRA'. Use the total intra_movement of the month as the montly mobility.",
#                  ]


# task = "Generate a graph (data structure) only, whose nodes are (1) a series of consecutive steps and (2) data to solve this question: Plot all the points with LC0_Desc=Woodland in Europe. Save the result as a png. Use geopandas. Don't join these shapefiles just plot them. use geopandas. save the result as a png."
# data_locations_str = ["GeoDataFrame: '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'."
# "Europe shapefile: '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'."]

# task = "plot all the points with pH_CaCl2<6 in Europe. Save the result as a png. Use geopandas."

# data_locations_str = ["You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'.",
# "Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'."
# ]


G, extract = generate_graph(pipe, task, data_locations_str)

# extract = """"""

# exec(extract)

operations = []
codes = []
for node in G.nodes():
    if G.nodes[node]['node_type'] == "operation":
        name = node
        description = G.nodes[node]['description']
        args = ",".join(list(G.predecessors(node)))
        first_successor = list(G.successors(node))[0]
        return_description = G.nodes[first_successor]['description']
        operations.append({"name":name, "args":args})
        code = generate_code(model='', pipe=pipe, task=task, funcname=name, desc=description, args=args, returns=return_description, solution_graph=G, extract=extract, data_locations_str=data_locations_str)
        codes.append(code)



        exec(code)

assembly = generate_assembly_code(pipe, codes, task, data_locations_str)
# functioncall = globals()['assembely_solution']

# assembly_solution()



