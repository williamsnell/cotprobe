def gen_prompt(question):
    options = "\n".join([f"{chr(ord('A') + i)}: {option}" for i, option in enumerate(question['options'])])
    
    return f"""Answer the following multiple choice question, selecting \
from the answer A through to J. After thinking, reply \
directly with your answer. 

The last character of your response must be the letter you have chosen.

Question:

{question['question']}

Options:

{options}

Answer:
    <think>\n"""
