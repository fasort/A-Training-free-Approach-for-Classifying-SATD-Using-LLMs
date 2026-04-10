import pandas as pd
import random
import re
import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = LLM(model="/root/xinglin-data/satd/models/LLM-Research/gemma-2-27b-it",gpu_memory_utilization=0.9)


generation_config = SamplingParams(temperature=0.7, top_p = 0.95,  max_tokens = 200)

def generate_prompt(comment_text, is_satd, satd_type, file_content, commit_message):
    prompt = f'''<bos><start_of_turn>user
    There are six types of software technical debts:

    Requirement: Requirement debts can be functional or non-functional. In the functional case, implementations are left unfinished or in need of future feature support. In the non-functional case, the corresponding code does not meet the requirement standards (speed, memory usage, security, etc...).

    Code: Bad coding practices leading to poor legibility of code, making it difficult to understand and maintain.

    M&T: Problems found in implementations involving testing or monitoring subcomponents.

    Defect: Identified defects in the system that should be addressed.

    Design: Areas which violate good software design practices, causing poor flexibility to evolving business needs.

    Documentation: Inadequate documentation that exists within the software system. 

    Below is a code comment and its corresponding label. Provide an explanation of why the comment is labeled as "{satd_type}":

    - Code Comment: "{comment_text}"
    - Label: "{satd_type}"

    Ensure your response is short and concise, within one paragraph and less than 100 tokens.<end_of_turn>\n<start_of_turn>model\n
    '''
    return prompt

file_path = "/root/xinglin-data/satd/Dataset/23_Shades/OBrien_789_v2.csv"
# file_path = "/root/xinglin-data/projects/deep_learning_experiment/data/raw/Dataset/Maldonado-62k/not_corrected_800.csv"

df = pd.read_csv(file_path)

comment_text = df['comment_text']
is_satd = df['is_satd']
satd_type = df['satd_type']
file_content = df['file_content']
commit_message = df['commit_message']


def get_response(model, generation_config, prompt):

    outputs = model.generate(prompt, generation_config)
 

    return outputs[0].outputs[0].text

def extract_satd_from_response(response):
    # 提取response中的SATD或Not-SATD
    # satd = response.split('\n')[-1].split('.')[1].strip()
    return response

# 用于存储新的SATD列数据
new_satd_values = []

for comment_text, is_satd, satd_type, file_content, commit_message in zip(comment_text, is_satd, satd_type, file_content, commit_message):
    prompt = generate_prompt(comment_text, is_satd, satd_type, file_content, commit_message)
    response = get_response(model, generation_config, prompt)
    satd_value = extract_satd_from_response(response)
    new_satd_values.append(satd_value)
    print(satd_value)

# 将提取的exp值更新到原数据集中
df['exp'] = new_satd_values

# 保存更新后的数据集到新文件
output_file_path = "/root/xinglin-data/satd/newData/OBrien_with_exp_gemma_only_comment_100.csv"
df.to_csv(output_file_path, index=False)