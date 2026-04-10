import pandas as pd
import random
import re
import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = LLM(model="/root/xinglin-data/satd/models/LLM-Research/gemma-2-27b-it",gpu_memory_utilization=0.9)


generation_config = SamplingParams(temperature=0.7, top_p = 0.95,  max_tokens = 200)

# add context and prompt_context colums to the dataset
def add_context(df, inputs):
    context = [] # to be used in SentenceTransform
    prompt_context = [] # to be used in prompt generation
    for _,row in df.iterrows():
        if inputs == 'ct':
            if DATASET=='Maldonado62k':
                context.append(row['comment_text'])
                prompt_context.append('### Comment text: """ ' + row['comment_text'] + ' """')
            else:
                context.append(row['comment_text'])
                prompt_context.append('### Technical debt comment: """ ' + row['comment_text'] + ' """')
        elif inputs == 'fp+ct':
            context.append(row['file_path'] + '\n' + row['comment_text'])
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """')
        elif inputs == 'fp+cms+ct':
            context.append(row['file_path'] + '\n' + str(row['containing_method_signature']) + '\n' + row['comment_text'])
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Containing method signature: """ ' + str(row['containing_method_signature']) + ' """\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """')
        elif inputs == 'fp+ct+cmb':
            context.append(row['file_path'] + '\n' + row['comment_text'] + '\n' + str(row['containing_method']))
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """\n' +
                       '### Containing method: """ ' + str(row['containing_method']).replace('"""',"'''") + ' """')
        else:
            print('ERROR!')

    df['context'] = context
    df['prompt_context'] = prompt_context
    return df

def generate_prompt(prompt_context, satd_type):
    prompt = f'''<bos><start_of_turn>user
    There are six types of software technical debts:

    Requirement: Requirement debts can be functional or non-functional. In the functional case, implementations are left unfinished or in need of future feature support. In the non-functional case, the corresponding code does not meet the requirement standards (speed, memory usage, security, etc...).

    Code: Bad coding practices leading to poor legibility of code, making it difficult to understand and maintain.

    M&T: Problems found in implementations involving testing or monitoring subcomponents.

    Defect: Identified defects in the system that should be addressed.

    Design: Areas which violate good software design practices, causing poor flexibility to evolving business needs.

    Documentation: Inadequate documentation that exists within the software system. 

    Below is a technical debt and its corresponding label. Provide an explanation of why the technical debt is labeled as "{satd_type}":

    {prompt_context}

    Ensure your response is short and concise, within one paragraph and less than 100 tokens.<end_of_turn>\n<start_of_turn>model\n
    '''
    return prompt




def get_response(model, generation_config, prompt):

    outputs = model.generate(prompt, generation_config)
 

    return outputs[0].outputs[0].text

def extract_satd_from_response(response):
    # 提取response中的SATD或Not-SATD
    # satd = response.split('\n')[-1].split('.')[1].strip()
    return response.strip()



file_path = "/root/xinglin-data/satd/Dataset/23_Shades/OBrien_789_v2.csv"

data = pd.read_csv(file_path)
data = data.rename(columns={"filename": "file_path"})
DATASET = 'OBrien'
inputs_list = ['fp+ct', 'ct', 'fp+cms+ct', 'fp+ct+cmb']


for INPUT in inputs_list:

    df = add_context(data, INPUT)

    satd_type = df['satd_type']
    prompt_context = df['prompt_context']

    # 用于存储新的SATD列数据
    new_satd_values = []

    for prompt_context, satd_type in zip(prompt_context, satd_type):
        prompt = generate_prompt(prompt_context, satd_type)
        print(prompt)
        response = get_response(model,generation_config,prompt)
        satd_value = extract_satd_from_response(response)
        new_satd_values.append(satd_value)
        print(satd_value)

    # 将提取的exp值更新到原数据集中
    df['exp'] = new_satd_values

    # 保存更新后的数据集到新文件
    output_file_path = f"/root/xinglin-data/satd/newData/OBrien_with_exp_gemma_{INPUT}.csv"
    df.to_csv(output_file_path, index=False)