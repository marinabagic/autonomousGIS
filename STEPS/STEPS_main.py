import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import logging
import os
from STEPS_generation import generate_steps
from STEPS_implementation import implement_steps
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('transformers').setLevel(logging.ERROR)


# model_name, kode = "mistralai/Mistral-7B-Instruct-v0.2", mistral2
# model_name, kode = "HuggingFaceH4/zephyr-7b-beta", zephyr
# model_name, kode = "teknium/OpenHermes-2.5-Mistral-7B", openhermes
# model_name, kode = "upstage/SOLAR-10.7B-Instruct-v1.0", solar
# model_name, kode = "mistralai/Mistral-7B-Instruct-v0.3", mistral3
# model_name, kode = "meta-llama/Meta-Llama-3-8B-Instruct", llama
# model_name, kode = "gradientai/Llama-3-8B-Instruct-Gradient-1048k", "gradient"
model_name, kode = "bigcode/starcoder2-15b-instruct-v0.1", "starcoder"

file_path = "/home/fkriskov/diplomski/datasets/lucas-soil-2018-v2/lucas-soil-2018 copy.csv"
europe = gpd.read_file("/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp")
df = pd.read_csv(file_path)
df_str = df.head()
# codes = list(set(df['NUTS_0']))
columns = list(df.columns)
europe_columns = europe.columns

pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
file_paths = ["You are working with a CSV file that is located in '/home/fkriskov/diplomski/datasets/lucas-soil-2018-v2/lucas-soil-2018 copy.csv'.\n\nThese are the columns of the dataframe:{columns}\n\nThis is the head of the dataframe:{df_str}\n"]
# file_paths = [f"You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'.\n\nThese are the columns of the dataframe:{columns}\n\nThis is the head of the dataframe:{df_str}\n",
# f"Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'.\n\nThese are the columns of the Europe Shapefile:{europe_columns}\n"]
# europe = gpd.read_file("/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp")


# "You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'."
# "Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'."

obj, filename = "Which land type (LC0_Desc) has the highest average 'pH_H2O'.", "desc_1_"

print("############# STEP GENERATION #############")
steps = generate_steps(pipe, file_paths, obj, [])

print("############# CODE IMPLEMENTATION #############")
code = implement_steps(pipe, file_paths, obj, steps, [])

# with open(os.path.join("/home/fkriskov/diplomski/testing/uspjesni-rezultati", kode, filename+kode+".txt"), 'w') as f:
#     f.write(code)

# User Loop
while True:
    print("############# USER LOOP #############")
    oldmessages = [
            {
                "role": "user",
                "content": f"""
    "for these files: {"".join(file_paths)}"
    "answers this user query:{obj}"
    """
            },
            {
                "role": "assistant",
                "content": f"""
    {steps}

    {code}
    """
            },
        ]

    obj = input("Is there something else you want to ask? Ask away: ")
    if obj == "EXIT": break

    print("############# STEP GENERATION #############")
    steps = generate_steps(pipe, file_paths, obj, oldmessages)

    print("############# CODE IMPLEMENTATION #############")
    code = implement_steps(pipe, file_paths, obj, steps, oldmessages)