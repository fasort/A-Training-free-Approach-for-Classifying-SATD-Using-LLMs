import pandas as pd
import random
import re
import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = LLM(model="/root/xinglin-data/satd/models/LLM-Research/gemma-2-27b-it",gpu_memory_utilization=0.9)


generation_config = SamplingParams(temperature=0.7, top_p = 0.95,  max_tokens = 200)

def generate_prompt(comment_text, satd_type):
    if satd_type == 'IMPLEMENTATION':
        satd_type = 'REQUIREMENT'
    prompt = f'''<bos><start_of_turn>user
    There are five types of software technical debts:

    REQUIREMENT: Requirement debt comments express incompleteness of the method, class or program as observed in the following comments:
    “/TODO no methods yet for getClassname” - [from Apache Ant]
    “//TODO no method for newInstance using a reverse-classloader” - [from Apache Ant]
    “TODO: The copy function is not yet * completely implemented - so we will * have some exceptions here and there.*/” - [from ArgoUml]


    TEST: Test debt comments are the ones that express the need for implementation or improvement of the current tests. As shown in the examples below, test debt comments are very straight forward in their meaning.
    “// TODO - need a lot more tests” - [from Apache Jmeter]
    “//TODO enable some proper tests!!” - [from Apache Jmeter]


    DEFECT: Self-admitted defect debt: In defect debt comments the author states that a part of the code does not have the expected behavior, meaning that there is a defect in the code.
    “// Bug in above method” - [from Apache Jmeter]
    “// WARNING: the OutputStream version of this doesn’t work!” - [from ArgoUml]
    As shown in these examples there are defects that are known by the developers, but for some reason is not fixed yet.


    DESIGN: These comments indicate that there is a problem with the design of the code. They can be comments about misplaced code, lack of abstraction, long methods, poor implementation, workarounds or a temporary solution. Lets consider the following comments: 
    “TODO: - This method is too complex, lets break it up” - [from ArgoUml]
    “/* TODO: really should be a separate class */” -[from ArgoUml]
    These comments are clear examples of what we consider as self-admitted design debt.


    DOCUMENTATION: In the documentation debt comments the author express that there is no proper documentation supporting that part of the program.
    “**FIXME** This function needs documentation” -[from Columba]
    “// TODO Document the reason for this” - [from Apache Jmeter]
    Here, the developers clearly recognize the need to document their code, however, for some reason they do not document it yet.


    Below is a code comment and its corresponding label. Provide an explanation of why the comment is labeled as "{satd_type}":

    - Code Comment: "{comment_text}"
    - Label: "{satd_type}"

    Ensure your response is short and concise, within one paragraph and less than 100 tokens.<end_of_turn>\n<start_of_turn>model\n
    '''
    return prompt

file_path = "/root/xinglin-data/satd/newData/labeled_dataset.csv"
# file_path = "/root/xinglin-data/projects/deep_learning_experiment/data/raw/Dataset/Maldonado-62k/not_corrected_800.csv"

df = pd.read_csv(file_path)

comment_text = df['Comment']
satd_type = df['Category']
project = df['Project']


def get_response(model, generation_config, prompt):

    outputs = model.generate(prompt, generation_config)
 

    return outputs[0].outputs[0].text

def extract_satd_from_response(response):
    # 提取response中的SATD或Not-SATD
    # satd = response.split('\n')[-1].split('.')[1].strip()
    return response

# 用于存储新的SATD列数据
new_satd_values = []

for project, comment_text, satd_type in zip(project, comment_text, satd_type):
    prompt = generate_prompt(comment_text, satd_type)
    response = get_response(model, generation_config, prompt)
    satd_value = extract_satd_from_response(response)
    new_satd_values.append(satd_value)
    print(satd_value)

# 将提取的exp值更新到原数据集中
df['exp'] = new_satd_values

# 保存更新后的数据集到新文件
output_file_path = "/root/xinglin-data/satd/newData/New_data_with_exp_gemma_100.csv"
df.to_csv(output_file_path, index=False)