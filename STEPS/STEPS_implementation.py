
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


def implement_steps(pipe, file_paths, user_query, steps, messages):
    instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The solution should be a Python expression that can be called with the `exec()` function, inside '```python ```'.\n"
    "3. The code should represent a solution to the query.\n"
    "4. If not instructed otherwise, print the final result variable.\n"
    "5. If you are asked to plot something, save it as a plot.png.\n"
    "6. Don't explain the code.\n"
    )

# "You cannot merge these shapefiles,just plot them."
# "set marker='.' and figsize (10,10)"

    messages += [
        {
            "role": "user",
            "content": f"""
"files needed are {"".join(file_paths)}"

"'NUTS_0' is Alpha-2 code."

"for the given file locations and for these solution steps: {steps}"

"generate a complete python code that follows these steps and answers this user query:{user_query}"

{instruction_str}
"""
# "The possible NUTS_0 codes are: {codes}"
        },
    ]

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

    
    # double check
    checked_code = double_check(pipe, python_extract, messages)
    # print(checked_code)
    try:
        error_occured = False
        print("\nRunning Code...\n")
        exec(checked_code)
    except Exception as e:
        error_occured = True
        error = e
        print("\nerror occured: ", e, "\n")


    # error code fixer
    python_fix_extract = checked_code
    while error_occured:
        python_fix_extract = error_code_fixer(pipe, python_fix_extract, error)

        print("\n------------------FIXED!!!------------------\n")
        print(python_fix_extract)
        print("\n------------------REZULTAT!!!------------------\n")
        try:
            error_occured = False
            exec(python_fix_extract)
        except Exception as e:
            error_occured = True
            error = e
            print("\nerror occured: ", e, "\n")

    return python_fix_extract