# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor
import pandas as pd
import numpy as np
from error_code_fixer import error_code_fixer, double_check
import geopandas as gpd
from shapely.geometry import Point
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('transformers').setLevel(logging.ERROR)

models = {}

# model_name, kode = "mistralai/Mistral-7B-Instruct-v0.2", 'mistral2'
# models[kode] = model_name
# model_name, kode = "HuggingFaceH4/zephyr-7b-beta", 'zephyr'
# models[kode] = model_name
# model_name, kode = "teknium/OpenHermes-2.5-Mistral-7B", 'openhermes'
# models[kode] = model_name
# model_name, kode = "upstage/SOLAR-10.7B-Instruct-v1.0", 'solar'
# models[kode] = model_name
# model_name, kode = "mistralai/Mistral-7B-Instruct-v0.3", 'mistral3'
# models[kode] = model_name
# model_name, kode = "meta-llama/Meta-Llama-3-8B-Instruct", 'llama'
# models[kode] = model_name
# model_name, kode = "gradientai/Llama-3-8B-Instruct-Gradient-1048k", "gradient"
# models[kode] = model_name
model_name, kode = "bigcode/starcoder2-15b-instruct-v0.1", "starcoder"
models[kode] = model_name


# pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto", token="hf_XXXXXXX")
# file_path = "/home/fkriskov/diplomski/datasets/IRIS flower dataset.csv"
file_path = "/home/fkriskov/diplomski/datasets/lucas-soil-2018-v2/lucas-soil-2018 copy.csv"
europe = gpd.read_file("/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp")
df = pd.read_csv(file_path)
# geometry = [Point(xy) for xy in zip(df['TH_LONG'], df['TH_LAT'])]
# geo_df = gpd.GeoDataFrame(df, geometry=geometry)
# geo_df.set_crs(epsg=4326, inplace=True)
# if europe.crs != geo_df.crs:
#     geo_df = geo_df.to_crs(europe.crs)
# geo_df.to_file('/home/fkriskov/diplomski/datasets/geo_dataframe.shp')

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The solution should be a Python expression that can be called with the `exec()` function, inside '```python ```'.\n"
    "3. The code should represent a solution to the query.\n"
    "4. If not instructed otherwise, print the final result variable.\n"
    "5. If you are asked to plot something, save it as a plot.png.\n"
    "6. Don't explain the code.\n"
)


df_str = df.head()
codes = list(set(df['NUTS_0']))
columns = list(df.columns)
europe_columns = europe.columns

# user_query, filename = "Which land type (LC0_Desc) has the highest average 'pH_H2O'.", "desc_1_"
# user_query, filename = "Plot the average 'OC' for each land type (LC0_Desc). save it as a png", "desc_2_"
# user_query, filename = "Is there a significant relationship between land type (LC0_Desc) and pH_H2O? Use chi square from scipy.", "infer_1_"
# user_query, filename = "Perform a t-test to compare 'K' between Grassland and Cropland.", "infer_4_"
# user_query, filename = "Plot a linear regression analysis to see the relationship between 'pH_H2O' and 'K'. save it as a png", "infer_5_"
# user_query, filename = "Construct a 95% \confidence interval for the mean 'OC' content in the dataset.", "infer_6_"
# user_query, filename = "Using the Central Limit Theorem, simulate the sampling distribution of the mean 'pH_H2O' for sample sizes of 30. Plot the distribution and compare it to the normal distribution.", "infer_7_"
# user_query, filename = "Calculate the z-scores for 'EC' and identify any outliers (z-score > 3 or < -3). use zscore from scipy", "infer_8_"
# user_query, filename = "Perform a hypothesis test to determine if the mean 'K' content in the entire dataset is significantly different from 2%. Use a t-test for the hypothesis test.", "infer_9_"
# user_query, filename = "Calculate the p-value for the correlation between 'P' and 'K'. Determine if the correlation is statistically significant.", "infer_10_"
# user_query, filename = "Plot the points with 'pH_H2O'>5 blue and 'pH_H2O'<5 red in Europe. save it as a png", "geo_8_"
# user_query, filename = "Create a map displaying the distribution of soil types ('LC0_Desc') across Europe. Each soil type should be represented by a different color. Use geopandas and save the map as a png.", "geo_9_"
# user_query, filename = "Plot all the LC0_Desc='Grassland' and LC0_Desc='Woodland' points where 'OC'>20. Use geopandas and save the map as a png.", "geo_10_"


# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
#"You are working with a CSV file that is located in {file_path}"
# "You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'."
# "Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'."
# "You cannot merge these shapefiles,just plot them."
# "set marker='.' and figsize (10,10)"
# "These are the columns of the dataframe:"
# {columns}
# "These are the columns of the Europe Shapefile:"
# {europe_columns}

messages = [
    {
        "role": "user",
        "content": f"""
"You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'."
"Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'."
"You cannot merge these shapefiles,just plot them."
"set marker='.' and figsize (10,10)"
"These are the columns of the dataframe:"
{columns}
"These are the columns of the Europe Shapefile:"
{europe_columns}

"'NUTS_0' is Alpha-2 code."
"The possible NUTS_0 codes are: {codes}"
"And this is the head of the dataframe:"
"{df_str}\n"
"Follow these instructions:"
"{instruction_str}"

"answer this query:"
"{user_query}\n"
"""
    },
    # desc_1
    # {"role": "user", "content": "What is the average petal length of top 30 longest sepals?"},  # 5.623333333333333
    # infer_1
    # {"role": "user", "content": "Is there a significant relationship between species and sepal length? Use chi square from scipy."},
    # desc_2
    # {"role": "user", "content": "Plot the average petal length for each iris species."},
    # desc_3
    # {"role": "user", "content": "Calculate the average pH for south EU."},
    # desc_4
    # {"role": "user", "content": "Calculate the average pH for Austria, from the mentioned csv."},  #5.302227171492205
    # desc_5
    # {"role": "user", "content": "Calculate the max value of 'N' for Slovenia, from the mentioned csv."},
    # infer_2
    # {"role": "user", "content": "Is there a significant difference between 'N' in Austria and France? Use ANOVA from scipy."},
    # infer_3
    # {"role": "user", "content": "Which parameter has the strongest correlation with EC among {pH_CaCl2, pH_H2O, OC, CaCO3, P, N, K}?"},
    # no name
    # {"role": "user", "content": "plot linear regressions with EC and pH_CaCl2, pH_H2O, OC, CaCO3, P, N and K, respectively. Use sklearn."},
    # geo_1
    # {"role": "user", "content": "plot all the points that have pH_CaCl2 > 6. use geopandas. save the image as a png."},
    # geo_2
    # {"role": "user", "content": "plot all the points with LC0_Desc=Woodland in Europe. Save the result as a png. Use geopandas."},
    # scatter
    # {"role": "user", "content": "plot scatter of EC and pH_CaCl2 and save it as scatter.png."},
    # geo_3
    # {"role": "user", "content": "plot all the points with LC0_Desc=Woodland & pH<6 in Europe. Save the result as a png. Use geopandas."},
]
for model in models:
    model_name = models[model]
    kode = model
    # input(f"next model?, {kode}")
    pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto", token="hf_XXXXXXXX")

    invalid_output = True
    temp_messages = messages
    while invalid_output:
        prompt = pipe.tokenizer.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)
        result = outputs[0]["generated_text"]
        print(result)
        
        pythonbracketcounter = instruction_str.count('```python') + 1
        if result.count("```python") == pythonbracketcounter:
            # temp_messages = messages + ["Display only the complete Python solution.\n"]
            invalid_output = False

    python_extract = result.split("```python")[pythonbracketcounter].split("```")[0]
    print("\n------------------GREAT SUCCESS!!!------------------\n")
    print(python_extract)
    print("\n------------------REZULTAT!!!------------------\n")

    kodename = kode
    if kode.startswith("mistral"): kodename="mistral"
    # with open(os.path.join("/home/fkriskov/diplomski/testing/uspjesni-rezultati", kode, filename+kodename+".txt"), 'w') as f:
    #     print("WRITING!", filename+kode+".txt")
    #     f.write(python_extract)

    try:
        error_occured = False
        exec(python_extract)
    except Exception as e:
        error_occured = True
        error = e
        print("error occured: ", e)

# double check
# checked_code = double_check(pipe, python_extract, messages)
# # print(checked_code)
# try:
#     error_occured = False
#     print("\nRunning Code...\n")
#     exec(checked_code)
# except Exception as e:
#     error_occured = True
#     error = e
#     print("\nerror occured: ", e, "\n")


# error code fixer
# python_fix_extract = checked_code
# while error_occured:
#     python_fix_extract = error_code_fixer(pipe, python_fix_extract, error)

#     print("\n------------------FIXED!!!------------------\n")
#     print(python_fix_extract)
#     print("\n------------------REZULTAT!!!------------------\n")
#     try:
#         error_occured = False
#         exec(python_fix_extract)
#     except Exception as e:
#         error_occured = True
#         error = e
#         print("\nerror occured: ", e, "\n")
