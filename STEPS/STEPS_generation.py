import torch
from transformers import pipeline
import pandas as pd


def generate_steps(pipe, file_paths, obj, messages):

    messages += [
        {
            "role": "user",
            "content": f"""For the given objective: {obj}
and these files:
{"".join(file_paths)}
come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. \
Steps should be clearly noted by having 'Step X:' written before the step itself, where X is the step number. \
DO NOT WRITE ANY CODE!!!!!
"""
        },
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
    result = outputs[0]["generated_text"]
    print(result)

    finalres = ""
    remove_python = result.split("```python")
    for each in remove_python:
        splited = each.split("```")
        each = splited[len(splited)-1]
        finalres += each

    print(finalres)
