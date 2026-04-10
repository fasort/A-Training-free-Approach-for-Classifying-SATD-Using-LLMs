from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from sentence_transformers import SentenceTransformer
import sentence_transformers
import torch
import torch.nn as nn
import pandas as pd
import gc
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import re
import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# add context and prompt_context colums to the dataset
def add_context(df, inputs):
    context = [] # to be used in SentenceTransform
    prompt_context = [] # to be used in prompt generation
    for _,row in df.iterrows():
        if inputs == 'ct':
            context.append(row['Comment'])
            prompt_context.append('### Technical debt comment: """ ' + row['Comment'] + ' """')
        else:
            print('ERROR!')

    df['context'] = context
    df['prompt_context'] = prompt_context
    return df

# read and prepare the OBrien dataset (SATD classification)

# df = pd.read_csv('Dataset/23_Shades/OBrien_789.csv') # this version doesn't have the containing_method and containing_method_signature columns
# df = pd.read_csv('/root/xinglin-data/satd/newData/New_data_with_exp_zhipu_150.csv')
df = pd.read_csv('/root/xinglin-data/satd/newData/New_data_with_exp_qwen32_100.csv')
print(len(df))
print(df.columns)


DATASET = 'New'

INPUT = 'ct'               # only comment text

df = add_context(df, INPUT)

print('\n-------------- An example of input data - context ---------------\n')
print(df.context[2]) # 151
print('\n-------------- An example of input data - prompt_context ---------------\n')
print(df.prompt_context[2]) # 151
print('-------------------------------------------------------\n')

df = df[['context','prompt_context','Category','Project','exp']]

# for each project, split data to train and test and save it in a dataset
dataset = {}
for test_fold in sorted(set(df['Project'])):
    test_df  = df[df['Project'] == test_fold]
    train_df = df[df['Project'] != test_fold]        
    train_df = train_df.sample(frac=1, random_state=42) # shuffle train

    data = DatasetDict({"train": Dataset.from_pandas(train_df), "test": Dataset.from_pandas(test_df)})        
    data=data.rename_column("Category","label")
    data=data.remove_columns(['Project','__index_level_0__'])
    dataset[test_fold] = data
    
METRIC = 'accuracy'


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import pandas as pd
import numpy as np

checkpoint = "/root/xinglin-data/satd/models/LLM-Research/codegemma-1.1-7b-it"

model = LLM(model="/root/xinglin-data/satd/models/LLM-Research/codegemma-1.1-7b-it",gpu_memory_utilization=0.9)


# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# generation_config = GenerationConfig(max_new_tokens=5, do_sample=True, temperature=0.01) # when I set max_new_tokens=3, it generates 'Require' rather 'Requirement'

# gemma
generation_config = SamplingParams(temperature=0.7, top_p = 0.95, max_tokens = 4096)

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""
### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""





def get_response(model, generation_config, prompt):

    outputs = model.generate(prompt, generation_config)
 

    return outputs[0].outputs[0].text


def get_confmat_str(real, pred, labels):
    output = ''
    cm = confusion_matrix(real, pred, labels=labels)
    max_label_length = max([len(label) for label in labels] + [5])
    output = " " * max_label_length + " " + " ".join(label.ljust(max_label_length) for label in labels) + "\n"
    for i, label in enumerate(labels):
        row = " ".join([str(cm[i][j]).ljust(max_label_length) for j in range(len(labels))])
        output += label.ljust(max_label_length) + " " + row + "\n"
    return output

def split_to_tokens(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def get_the_most_relevant_items_for_an_item(item_embed, items_embed, n):
    cos_sim = sentence_transformers.util.cos_sim(item_embed, items_embed) # Note: make sure to pass ndarray not list due to performance
    itemId_similarity = dict(zip(range(len(items_embed)),cos_sim.tolist()[0]))
    itemId_similarity = dict(sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)) # sort
    itemId_similarity = [(k,itemId_similarity[k]) for k in list(itemId_similarity)[:n]] # take top n
    return itemId_similarity

def get_the_most_relevant_items_for_an_item_given_cos_sim(item_indx, cos_sim, n):
    cos_sim_row = cos_sim[item_indx]
    itemId_similarity = dict(zip(range(len(cos_sim_row)),cos_sim_row.tolist()))
    itemId_similarity = dict(sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)) # sort
    itemId_similarity = [(k,itemId_similarity[k]) for k in list(itemId_similarity)[:n]] # take top n
    return itemId_similarity

init_prompt_for_OBrien =  \
'''<bos><start_of_turn>user
There are five types of software technical debts:

IMPLEMENTATION: Implementation debt comments express incompleteness of the method, class or program.

TEST: Test debt comments are the ones that express the need for implementation or improvement of the current tests.

DEFECT: In defect debt comments the author states that a part of the code does not have the expected behavior, meaning that there is a defect in the code.

DESIGN: These comments indicate that there is a problem with the design of the code. They can be comments about misplaced code, lack of abstraction, long methods, poor implementation, workarounds or a temporary solution. 

DOCUMENTATION: In the documentation debt comments the author express that there is no proper documentation supporting that part of the program.


Here are some examples:\n\n'''


def generate_prompt_without_adding_dynamic_examples(init_prompt, test_context):
    prompt = init_prompt
    prompt += test_context + '\n'
    prompt += '### Label: '
    return prompt

def generate_prompt_by_top_n_items(init_prompt, test_context, top_n_items, data):
    prompt = init_prompt
    for indx,similarity in top_n_items:
        if len(split_to_tokens(prompt+data['prompt_context'][indx]+test_context))<4096:
            prompt += data['prompt_context'][indx] + '\n'
            prompt += '### Label: ' + data['label'][indx] + '\n'
            prompt += '### Explanation: ' + data['exp'][indx].strip() + '\n\n'

            # label down
            # prompt += '### Explanation: ' + data['exp'][indx].strip() + '\n'
            # prompt += '### Label: ' + data['label'][indx] + '\n\n'

    prompt += 'Give the type of following software technical debt:\n'
    prompt += test_context + '\n'
    prompt += 'You only need answer the type.<end_of_turn>\n<start_of_turn>model\n'
    return prompt

def generate_prompt_by_random_n_items(init_prompt, test_context, num_rand, data):
    random_n_items = random.sample(range(len(data)), num_rand)
    prompt = init_prompt
    for indx in random_n_items:
        if len(split_to_tokens(prompt+data['prompt_context'][indx]+test_context))<500:
            prompt += data['prompt_context'][indx] + '\n'
            prompt += '### Label: ' + data['label'][indx] + '\n\n'
    prompt += test_context + '\n'
    prompt += '### Label: '
    return prompt

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from datasets import load_metric

# ICL_METHOD = 'task-level' # use the same prompt (zero-shot or the same demonstration examples) for all test data
ICL_METHOD = 'instance-level-nearest' # use different prompts (selects different demonstration examples) for different test samples by nearest examples selection
# ICL_METHOD = 'instance-level-random' # use different prompts (selects different demonstration examples) for different test samples by random selection



if DATASET == 'New':
    project_name = 'apache-ant-1.7.0'
    # indx = 11
    INIT_PROMPT = init_prompt_for_OBrien
    # INIT_PROMPT = "" # provide no description for the task (i.e., just provide some examples)
elif DATASET == 'Maldonado62k':
    # INIT_PROMPT = init_prompt_for_Maldonado62k # include no keywords
    INIT_PROMPT = init_prompt_for_Maldonado62k_MAT # include MAT keywords
    # INIT_PROMPT = init_prompt_for_Maldonado62k_Easy # include Easy keywords
    # INIT_PROMPT = init_prompt_for_Maldonado62k_GPT4 # include GPT4 keywords
else:
    print("ERROR! Unknown dataset")

random.seed(42)

st_model = SentenceTransformer('all-MiniLM-L6-v2') # model size: 80MB

if 'instance' in ICL_METHOD:
    if len(INIT_PROMPT)>0:
        num_instances = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    else:
        num_instances = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
else:
    num_instances = [0]

for NUM_EXAMPLES_IN_PROMPT in num_instances:
    print('Run the experiments for num_instances=',num_instances)

    if ICL_METHOD == 'task-level':
        icl_name = '_ICL-task'
    elif ICL_METHOD == 'instance-level-nearest':
        icl_name = '_ICL-nearest-' + str(NUM_EXAMPLES_IN_PROMPT).zfill(2)
    elif ICL_METHOD == 'instance-level-random':
        icl_name = '_ICL-random-' + str(NUM_EXAMPLES_IN_PROMPT).zfill(2)
    else:
        icl_name = '_ICL-error'

    print('Run the experiments for: ' + DATASET + '_Input-' + INPUT + '_' + checkpoint.split('/')[-1] + icl_name)
    # file_name = 'Results/' + 'labeldown' + DATASET + '_Input-exp-qwen-100-without-example_' + INPUT + '_' + checkpoint.split('/')[-1] + icl_name
    file_name = 'Results/' + 'labelforward' + DATASET + '_Input-exp-qwen-100-without-example_' + INPUT + '_' + checkpoint.split('/')[-1] + icl_name

    test_results = {}
    projects_real = {}
    projects_pred = {}
    all_real = []
    all_pred = []
    all_context = []
    all_project = []
    labels = list(set(dataset[project_name]['train']['label']))

    unrecognized_pred = 0 # don't move it to outer loop
    with open(file_name+'_confmat.txt', "w") as output_file:
        for project_name, data in dataset.items():
            print('---------- '+str(project_name)+' ----------')
            output_file.write('\n---------- '+str(project_name)+' ----------\n')
            torch.cuda.empty_cache()
            gc.collect()
            if True: # ICL_METHOD == 'instance-level-nearest': 
                train_data_embed = st_model.encode(data['train']['context'], show_progress_bar=False)
                test_data_embed = st_model.encode(data['test']['context'], show_progress_bar=False)
                cos_sim = sentence_transformers.util.cos_sim(test_data_embed, train_data_embed)
            test_results[project_name] = []
            projects_real[project_name] = []
            projects_pred[project_name] = []
            for indx, row, row_embed in zip(range(len(data['test'])), data['test'], test_data_embed):
                if ICL_METHOD == 'task-level':
                    # prompt = generate_prompt(INIT_PROMPT, row['prompt_context'])
                    prompt = generate_prompt_without_adding_dynamic_examples(INIT_PROMPT, row['prompt_context'])
                elif ICL_METHOD == 'instance-level-nearest':
                    top_n_items = get_the_most_relevant_items_for_an_item_given_cos_sim(indx, cos_sim, NUM_EXAMPLES_IN_PROMPT)
                    prompt = generate_prompt_by_top_n_items(INIT_PROMPT, row['prompt_context'], top_n_items, data['train'])
                elif ICL_METHOD == 'instance-level-random':
                    prompt = generate_prompt_by_random_n_items(INIT_PROMPT, row['prompt_context'], NUM_EXAMPLES_IN_PROMPT, data['train'])
                else:
                    print('ERROR!')
                #print(len(split_to_tokens(prompt)))
                if len(split_to_tokens(prompt))<4096:
                    print(prompt)
                    pred = get_response(model, generation_config, prompt)
                    print(pred)
                    for label in labels:
                        if label.lower() in pred.lower():
                            pred = label
                            break
                else:
                    pred = ''
                for label in labels:
                    if len(pred)>0 and pred.lower() == label.lower():
                        pred = label
                if pred not in labels:
                    #print(pred)
                    if DATASET=='Maldonado62k':
                        pred = 'Not-SATD'
                    elif DATASET=='New':
                        pred = 'DESIGN'
                    unrecognized_pred += 1
                if pred=='SATD' and row['label']=='Not-SATD' and False:
                    print(prompt)
                    print('--------------------------')
                projects_real[project_name].append(row['label'])               
                projects_pred[project_name].append(pred)
                all_context.append(row['prompt_context'])
                all_project.append(project_name)
            all_real += projects_real[project_name]    
            all_pred += projects_pred[project_name]
            # print precision recall and F1 for this project
            print(classification_report(projects_real[project_name], projects_pred[project_name], zero_division=0, digits=3))
            output_file.write(classification_report(projects_real[project_name], projects_pred[project_name], zero_division=0, digits=3)+"\n")
            # print confusion matrix for this project
            confmat_str = get_confmat_str(projects_real[project_name], projects_pred[project_name], labels=labels)
            print(confmat_str)
            output_file.write(confmat_str)
        print('=========== Overall ==========')
        output_file.write('\n=========== Overall ==========\n')
        # print precision recall and F1 for all data
        print(classification_report(all_real, all_pred, zero_division=0, digits=3))
        output_file.write(classification_report(all_real, all_pred, zero_division=0, digits=3)+"\n")
        # print confusion matrix for all data
        confmat_str = get_confmat_str(all_real, all_pred, labels=labels)
        print(confmat_str)
        output_file.write(confmat_str)
        print('\nNumber of unrecognized predictions:', unrecognized_pred, '\nWe considered them as the majority class.')
        output_file.write('\nNumber of unrecognized predictions: '+str(unrecognized_pred)+'\nWe considered them as the majority class.\n')

    test_result_df = pd.DataFrame({'project': all_project, 'context':all_context, 'real': all_real, 'pred': all_pred})
    test_result_df.to_csv(file_name+'_pred.csv', index=False)