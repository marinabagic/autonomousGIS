from error_code_fixer import error_code_fixer
import networkx as nx


def get_ancestor_operations(node_name, solution_graph):
    ancestor_operation_names = []
    ancestor_node_names = nx.ancestors(solution_graph, node_name)
    # for ancestor_node_name in ancestor_node_names:
    ancestor_operation_names = [node_name for node_name in ancestor_node_names if node_name in operation_node_names(solution_graph)]

    ancestor_operation_nodes = []
    for oper in operation_node_names(solution_graph):
        oper_name = oper['node_name']
        if oper_name in ancestor_operation_names:
            ancestor_operation_nodes.append(oper)

    return ancestor_operation_nodes




def operation_node_names(solution_graph):
    opera_node_names = []
    assert solution_graph, "The Solution class instance has no solution graph. Please generate the graph"
    for node_name in solution_graph.nodes():
        node = solution_graph.nodes[node_name]
        node['node_name'] = node_name
        if node['node_type'] == 'operation':
            opera_node_names.append(node)#_name)
    return opera_node_names




def get_descendant_operations(solution_graph, node_name):
    descendant__operation_names = []
    descendant_node_names = nx.descendants(solution_graph, node_name)
    # for descendant_node_name in descendant_node_names:
    descendant__operation_names = [node_name for node_name in descendant_node_names if node_name in operation_node_names(solution_graph)]
    # descendant_codes = '\n'.join([oper['operation_code'] for oper in descendant_node_names])
    descendant_operation_nodes = []
    for oper in operation_node_names(solution_graph):
        oper_name = oper['node_name']
        if oper_name in descendant__operation_names:
            descendant_operation_nodes.append(oper)

    return descendant_operation_nodes




def get_descendant_operations_definition(descendant_operations):
    keys = ['node_name', 'description', 'function_definition', 'return_line']
    operation_def_list = []
    for node in descendant_operations:
        operation_def = {key: node[key] for key in keys}
        operation_def_list.append(str(operation_def))
    defs = '\n'.join(operation_def_list)
    return defs




def generate_code(model, pipe, task, funcname, desc, args, returns, solution_graph, extract, data_locations_str):

#     prompt = f'''\
# <|fim_prefix|>
# def {funcname}({args}):
#     """
#     Function description: {desc}
#     Returns: {returns}
#     """
#     <|fim_suffix|>
#     return result<|fim_middle|>\
    # '''

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # prompt_len = inputs["input_ids"].shape[-1]
    # outputs = model.generate(**inputs, max_new_tokens=350)
    # print(tokenizer.decode(outputs[0][prompt_len:]))
    # print(outputs[0]['generated_text'])

    # print("\n------------------REZULTAT!!!------------------\n")
    # output = tokenizer.decode(outputs[0][prompt_len:])
    # output = output[:output.index("<|file_separator|>")]

    # final = prompt[15:len(prompt)-18].replace("<|fim_suffix|>", output)

    operation_reply_exmaple = """
```python',
def Load_csv(tract_population_csv_url="https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/NC_tract_population.csv"):
# Description: Load a CSV file from a given URL
# tract_population_csv_url: Tract population CSV file URL
tract_population_df = pd.read_csv(tract_population_csv_url)
return tract_population_df
```
"""

    operation_requirement = [                         
                        'DO NOT change the given variable names and paths.',
                        'Put your reply into a Python code block(enclosed by ```python and ```), NO explanation or conversation outside the code block.',
                        'If using GeoPandas to load a zipped ESRI shapefile from a URL, the correct method is "gpd.read_file(URL)". DO NOT download and unzip the file.',
                        # "Generate descriptions for input and output arguments.",
                        "You need to receive the data from the functions, DO NOT load in the function if other functions have loaded the data and returned it in advance.",
                        # "Note module 'pandas' has no attribute or method of 'StringIO'",
                        "Use the latest Python modules and methods.",
                        "When doing spatial analysis, convert the involved spatial layers into the same map projection, if they are not in the same projection.",
                        # "DO NOT reproject or set spatial data(e.g., GeoPandas Dataframe) if only one layer involved.",
                        "Map projection conversion is only conducted for spatial data layers such as GeoDataFrame. DataFrame loaded from a CSV file does not have map projection information.",
                        "If join DataFrame and GeoDataFrame, using common columns, DO NOT convert DataFrame to GeoDataFrame.",
                        # "When joining tables, convert the involved columns to string type without leading zeros. ",
                        # "When doing spatial joins, remove the duplicates in the results. Or please think about whether it needs to be removed.",
                        # "If using colorbar for GeoPandas or Matplotlib visulization, set the colorbar's height or length as the same as the plot for better layout.",
                        "Graphs or maps need to show the unit, legend, or colorbar.",
                        "Keep the needed table columns for the further steps.",
                        "Remember the variable, column, and file names used in ancestor functions when using them, such as joining tables or calculating.",                        
                        # "When crawl the webpage context to ChatGPT, using Beautifulsoup to crawl the text only, not all the HTML file.",
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "If using GeoPandas for spatial joining, the arguements are: geopandas.sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', **kwargs), how: the type of join, default ‘inner’, means use intersection of keys from both dfs while retain only left_df geometry column. If 'how' is 'left': use keys from left_df; retain only left_df geometry column, and similarly when 'how' is 'right'. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attributes are the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",

                        "DO NOT use 'if __name__ == '__main__:' statement because this program needs to be executed by exec().",
                        "Use the Python built-in functions or attribute. If you do not remember, DO NOT make up fake ones, just use alternative methods.",
                        "Pandas library has no attribute or method 'StringIO', so 'pd.compat.StringIO' is wrong, you need to use 'io.StringIO' instead.",
                        "Before using Pandas or GeoPandas columns for further processing (e.g. join or calculation), drop recoreds with NaN cells in those columns, e.g., df.dropna(subset=['XX', 'YY']).",
                        "When read FIPS or GEOID columns from CSV files, read those columns as str or int, never as float.",
                        "FIPS or GEOID columns may be str type with leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12), or integer type without leading zeros. Thus, when joining they, you can convert the integer colum to str type with leading zeros to ensure the success.",
                        "If you need to make a map and the map size is not given, set the map size to 15*10 inches.",
                        ]

    pre_requirements = [
            f'The function description is: {desc}',
            f'The function definition is: {funcname}({args})',
            f'The function return line is: {returns}'
        ]

    operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            pre_requirements + operation_requirement)])

    ancestor_operations = get_ancestor_operations(funcname, solution_graph)
    ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
    descendant_operations = get_descendant_operations(solution_graph, funcname)
    descendant_defs = get_descendant_operations_definition(descendant_operations)
    descendant_defs_str = str(descendant_defs)

    messages = [
    {
            "role": "user",
            "content": f"""
Your role: A professional Geo-information scientist and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You know well how to design and implement a function that meet the interface between other functions. Yor program is always robust, considering the various data circumstances, such as colum data types, avoiding mistakes when joining tables, and remove NAN cells before further processing. You have an good feeling of overview, meaning the functions in your program is coherent, and they connect to each other well, such as function names, parameters types, and the calling orders. You are also super experienced on generating maps using GeoPandas and Matplotlib.
operation_task: You need to generate a Python function to do: {desc}
This function is one step to solve the question/task: {task}
This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{extract}
Data locations: {data_locations_str}

Your reply needs to meet these requirements: \n {operation_requirement_str}
The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes}
The descendant function (if any) definitions for the question are (node_name is function name): \n {descendant_defs_str}
            """
        },
    ]
# Your reply example: {operation_reply_exmaple}
    invalid_output = True
    while invalid_output:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=750, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)
        result = outputs[0]["generated_text"]
        print(result)
        
        if result.count("```python") >= 2:
            invalid_output = False

    python_extract = result.split("```python")[-1].split("```")[0]
    print("\n------------------GREAT SUCCESS!!!------------------\n")
    print(python_extract)
    print("\n------------------REZULTAT!!!------------------\n")
    try:
        error_occured = False
        exec(python_extract)
        return python_extract
    except Exception as e:
        error_occured = True
        error = e
        print("error occured: ", e)


    # error code fixer
    python_fix_extract = python_extract
    while error_occured:
        python_fix_extract = error_code_fixer(pipe, python_fix_extract, error)

        print("\n------------------FIXED!!!------------------\n")
        print(python_fix_extract)
        print("\n------------------REZULTAT!!!------------------\n")
        try:
            error_occured = False
            exec(python_fix_extract)
            return python_fix_extract
        except Exception as e:
            error_occured = True
            error = e
            print("error occured: ", e)

    return



###################### ASSEMBLY ######################

def generate_assembly_code(pipe, codes, task, data_locations_str):
    assembly_role =  r'''A professional Geo-information scientist and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. Your are very good at assembling functions and small programs together. You know how to make programs robust.'''

    assembly_requirement = ['You can think step by step. ',
                        f"Each function is one step to solve the question. ",
                        f"The output of the final function is the question to the question.",
                        f"Put your reply in a code block(enclosed by ```python and ```), NO explanation or conversation outside the code block.",              
                        f"Save final maps, if any. If use matplotlib, the function is: matplotlib.pyplot.savefig(*args, **kwargs).",
                        f"The program is executable, put it in a function named 'assembly_solution()' then run it, but DO NOT use 'if __name__ == '__main__:' statement because this program needs to be executed by exec().",
                        "Use the built-in functions or attribute, if you do not remember, DO NOT make up fake ones, just use alternative methods.",
                        # "Drop rows with NaN cells, i.e., df.dropna(),  before using Pandas or GeoPandas columns for processing (e.g. join or calculation).",
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
                        ]
    all_operation_code_str = '\n'.join([operation for operation in codes])
    assembly_requirement = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(assembly_requirement)])
    assembly_prompt = f"Your role: {assembly_role} \n\n" + \
                        f"Your task is: use the given Python functions, return a complete Python program to solve the question: \n {task}" + \
                        f"Requirement: \n {assembly_requirement} \n\n" + \
                        f"Data location: \n {data_locations_str} \n" + \
                        f"Code: \n {all_operation_code_str}"
    messages = [
    {
            "role": "user",
            "content": assembly_prompt
        },
    ]

    invalid_output = True
    while invalid_output:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=2750, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)
        result = outputs[0]["generated_text"]
        print(result)
        
        if result.count("```python") >= 2:
            invalid_output = False

    python_extract = result.split("```python")[-1].split("```")[0]
    print("\n------------------GREAT SUCCESS!!!------------------\n")
    print(python_extract)
    print("\n------------------REZULTAT!!!------------------\n")
    try:
        error_occured = False
        exec(python_extract)
        return python_extract
    except Exception as e:
        error_occured = True
        error = e
        print("error occured: ", e)


    # error code fixer
    python_fix_extract = python_extract
    while error_occured:
        python_fix_extract = error_code_fixer(pipe, python_fix_extract, error)

        print("\n------------------FIXED!!!------------------\n")
        print(python_fix_extract)
        print("\n------------------REZULTAT!!!------------------\n")
        try:
            error_occured = False
            exec(python_fix_extract)
            return python_fix_extract
        except Exception as e:
            error_occured = True
            error = e
            print("error occured: ", e)

    return