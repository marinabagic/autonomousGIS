

def error_code_fixer(pipeline, code, error):
    print("\nRunning Error Code Fixer...\n")

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
    fixoutputs = pipeline(fixprompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
    fixresult = fixoutputs[0]["generated_text"]
    print(fixresult)

    if fixresult.count("```python") >= 1:
        python_fix_extract = fixresult.split("```python")[-1].split("```")[0]
    elif fixresult.count("```") >= 1:
        python_fix_extract = fixresult.split("```")[-2].split("```")[0]


    return python_fix_extract


def double_check(pipeline, code, messages):
    print("\nRunning Double Check...\n")

    checkmessages = [
    {
        "role": "user",
        "content": f"""You generated this code: {code}
based on these rules:
{messages[0]["content"]}

If there is something you think should be different, change it, if not, don't generate any code.
"""
    },
    # {"role": "user", "content": "Fix the code"}, 
    ]

    fixprompt = pipeline.tokenizer.apply_chat_template(checkmessages, tokenize=False, add_generation_prompt=True)
    fixoutputs = pipeline(fixprompt, max_new_tokens=512, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
    fixresult = fixoutputs[0]["generated_text"]

    if fixresult.count("```python") >= 1:
        python_fix_extract = fixresult.split("```python")[-1].split("```")[0]
    elif fixresult.count("```") >= 1:
        python_fix_extract = fixresult.split("```")[-2].split("```")[0]

    if(python_fix_extract == code):
        print("\nNo changes neccessary.\n")
        return code
    else:
        print(fixresult)
        return python_fix_extract