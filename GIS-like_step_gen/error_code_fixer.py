

def error_code_fixer(pipeline, code, error):
    messages = [
    {
        "role": "user",
        "content": f"""You generated this code: {code}
And this code gives out this error message: {error}
Fix the code
"""
    },
    # {"role": "user", "content": "Fix the code"}, 
    ]

    fixprompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    fixoutputs = pipeline(fixprompt, max_new_tokens=2750, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
    fixresult = fixoutputs[0]["generated_text"]
    print(fixresult)

    if fixresult.count("```python") == 1:
        python_fix_extract = fixresult.split("```python")[1].split("```")[0]
    elif fixresult.count("```") == 1:
        python_fix_extract = fixresult.split("```")[1].split("```")[0]


    return python_fix_extract