import torch
from transformers import pipeline
import pandas as pd
import numpy as np
import geopandas as gpd

def generate_graph(pipe, task, data_locations_str):
    # model_name = "HuggingFaceH4/zephyr-7b-beta"
    # model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto", token="hf_XXXXXX")

    data = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(data_locations_str)])

    messages = [
{
            "role": "user",
            "content": f"""
            Your role: A professional Geo-information scientist and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You know well how to set up workflows for spatial analysis tasks. You have significant experence on graph theory, application, and implementation. You are also experienced on generating map using Matplotlib and GeoPandas. You are also a specialist in Statistical Analysis.
            

            Your task: {task}
            

            Your reply needs to meet these requirements: 
            1. Think step by step.
            2. Steps and data (both input and output) form a graph stored in NetworkX. Disconnected components are NOT allowed.
            3. Each step is a data process operation: the input can be data paths or variables, and the output can be data paths or variables.
            4. There are two types of nodes: a) operation node, and b) data node (both input and output data). These nodes are also input nodes for the next operation node.
            5. The input of each operation is the output of the previous operations, except the those need to load data from a path or need to collect data.
            6. You need to carefully name the output data node, making they human readable but not to long.
            7. The data and operation form a graph.
            8. The first operations are data loading or collection, and the output of the last operation is the final answer to the task.Operation nodes need to connect via output data nodes, DO NOT connect the operation node directly.
            9. The node attributes include: 1) node_type (data or operation), 2) data_path (data node only, set to "" if not given ), and description. E.g., "name": “County boundary”, “data_type”: “data”, “data_path”: “D:\Test\county.shp”,  “description”: “County boundary for the study area”.
            10. The connection between a node and an operation node is an edge.
            11. Add all nodes and edges, including node attributes to a NetworkX instance, DO NOT change the attribute names.
            12. DO NOT generate code to implement the steps.
            13. Join the attribute to the vector layer via a common attribute if necessary.
            14. Put your reply into a Python code block, NO explanation or conversation outside the code block(enclosed by ```python and ```).
            15. Note that GraphML writer does not support class dict or list as data values.
            16. You need spatial data (e.g., vector or raster) to make a map.
            17. Do not put the GraphML writing process as a step in the graph.
            18. Keep the graph concise, DO NOT use too many operation nodes.
             

            Your reply example: 
            ```python
            import networkx as nx
            G = nx.DiGraph()
            # Add nodes and edges for the graph
            # 1 Load shapefile
            G.add_node("shp_url", node_type="data", path="https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/Hazardous_Waste_Sites.zip", description="shapefile URL")
            G.add_node("load_shp", node_type="operation", description="Load shapefile")
            G.add_edge("shp_url", "load_shp")
            G.add_node("gdf_file", node_type="data", description="GeoDataFrame")
            G.add_edge("load_shp", "gdf_file")
            ...
            ```
            

            Data locations (each data is a node):
            {data}
            """
        },
    ]
    invalid_output = True

    while invalid_output:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=1750, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)
        result = outputs[0]["generated_text"]
        print(result)
        
        if result.count("```python") == 3:
            invalid_output = False

        python_extract = result.split("```python")[3].split("```")[0]
        try:
            local_vars = {}
            exec(python_extract, {}, local_vars)
            G = local_vars['G']
        except (SyntaxError, TypeError, NameError) as e:
            print(e.msg)
            invalid_output = True
    return G, python_extract