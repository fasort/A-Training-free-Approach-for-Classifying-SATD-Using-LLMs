from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
import time
from sklearn.metrics import classification_report, confusion_matrix

# add the context column to a dataframe
def add_context(df, inputs):
    context = []
    for _,row in df.iterrows():
        if inputs == 'ct':
            context.append(row['comment_text'])
        elif inputs == 'fp+ct':
            context.append('file path: ' + str(row['file_path']) + '\n' +
                       'Technical debt comment: ' + row['comment_text'])
        elif inputs == 'fp+cms+ct':
            context.append('file path: ' + str(row['file_path']) + '\n' +
                       'Containing method signature: """ ' + str(row['containing_method_signature']) + ' """\n' +
                       'Technical debt comment: ' + row['comment_text'])
        elif inputs == 'fp+ct+cmb':
            context.append('file path: ' + str(row['file_path']) + '\n' +
                       'Technical debt comment: """ ' + row['comment_text'] + ' """\n' +
                       'Containing method body: """\n' + str(row['containing_method']).replace('"""',"'''") + '\n"""\n')
        else:
            print('ERROR!')

    df['context'] = context
    return df

df = pd.read_csv('/root/xinglin-data/satd/Dataset/23_Shades/OBrien_789_v2.csv')
print(len(df))
print(df.columns)

FRACTION = 1

# map labels
label_mapping = {
    'Requirement': 0,
    'Code': 1,
    'M&T': 2,
    'Defect': 3,
    'Design': 4,
    'Documentation': 5
}
df['satd_type'] = df['satd_type'].map(label_mapping)
df = df.rename(columns={"filename": "file_path"})

# INPUT = 'ct'               # only comment text
# INPUT = 'fp+ct'            # file path + comment text
# INPUT = 'fp+cms+ct'        # file path + containing method signature + comment text
INPUT = 'fp+ct+cmb'        # file path + comment text + containing method body

df = add_context(df, INPUT)

print('\n-------------- An example of input data ---------------\n')
print(df.context[150]) # 151 2 107 129 150 193
print('-------------------------------------------------------\n')

df = df[['context','satd_type','fold']]

# for each project, split data to train and test and save it in a dataset
dataset = {}
for test_fold in sorted(set(df['fold'])):
    if test_fold>0:
        valid_fold = test_fold - 1
    else:
        valid_fold = max(df['fold'])
    
    test_df  = df[df['fold'] == test_fold]
    train_df = df[df['fold'] != test_fold]        
    train_df = train_df.sample(frac=1, random_state=42) # shuffle train

    data = DatasetDict({"train": Dataset.from_pandas(train_df), "test": Dataset.from_pandas(test_df)})        
    data=data.rename_column("satd_type","label")
    data=data.remove_columns(['fold','__index_level_0__'])
    dataset[test_fold] = data
    
DATASET = 'OBrien'
TEXT_COLUMN = 'context'
LABEL_COLUMN = 'label'
NUM_LABELS = 6
METRIC = 'accuracy'

print(dataset)
print(df['satd_type'].value_counts())

# checkpoint = "bert-base-uncased"; HIDDEN_SIZE = 768;
# checkpoint = "microsoft/codebert-base"; HIDDEN_SIZE = 768;
# checkpoint = "google/flan-t5-small"; HIDDEN_SIZE = 512;
checkpoint = "/root/xinglin-data/satd/models/LLM-Research/flan-t5-base"; HIDDEN_SIZE = 768;
# checkpoint = "/root/xinglin-data/satd/models/LLM-Research/flan-t5-large"; HIDDEN_SIZE = 1024;
# checkpoint = "google/flan-t5-xl"; HIDDEN_SIZE = 2048;
# checkpoint = "roberta-base"; HIDDEN_SIZE = 768;
# checkpoint = "google/gemma-base"; HIDDEN_SIZE = 3072;

USE_LoRA = False
LOCAL_FILES_ONLY = False

if DATASET=='Maldonado62k':
    MAX_LEN = 128 
    if checkpoint=='bert-base-uncased':
        BATCH_SIZE = 32
        LR=0.00001
    elif checkpoint=='google/gemma-2b':
        BATCH_SIZE = 32
        LR=0.00001
    elif checkpoint=='roberta-base':
        BATCH_SIZE = 32
        LR=0.00001
    elif checkpoint=='microsoft/codebert-base':
        BATCH_SIZE = 32
        LR=0.00001
    elif checkpoint=='google/flan-t5-small':
        BATCH_SIZE = 32
        LR=0.0001
    elif checkpoint=='/root/xinglin-data/satd/models/LLM-Research/flan-t5-base':
        BATCH_SIZE = 32
        LR=0.0001
    elif checkpoint=='/root/xinglin-data/satd/models/LLM-Research/flan-t5-large':
        BATCH_SIZE = 16
        LR=0.0001
    elif checkpoint=='google/flan-t5-xl':
        BATCH_SIZE = 4
        LR=0.00002
elif DATASET=='OBrien':
    MAX_LEN = 512
    if checkpoint=='bert-base-uncased':
        BATCH_SIZE = 32
        LR=0.00005
    elif checkpoint=='microsoft/codebert-base':
        BATCH_SIZE = 32
        LR=0.00005
    elif checkpoint=='google/flan-t5-small':
        BATCH_SIZE = 32
        LR=0.001
    elif checkpoint=='/root/xinglin-data/satd/models/LLM-Research/flan-t5-base':
        BATCH_SIZE = 16
        LR=0.0005
    elif checkpoint=='/root/xinglin-data/satd/models/LLM-Research/flan-t5-large':
        BATCH_SIZE = 4
        LR=0.0002
    elif checkpoint=='google/flan-t5-xl':
        BATCH_SIZE = 1
        LR=0.00005
else:
    print('UNKNOWN DATASET!')
print('Dataset:',DATASET, '  MAX_LEN:', MAX_LEN, '  BATCH_SIZE:', BATCH_SIZE, '  USE_LoRA:', USE_LoRA, '  LR:', LR, '  Model:', checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
tokenizer.model_max_len = MAX_LEN



from transformers import T5Model
class CustomT5Model(T5Model):
    def forward(self, **kwargs):
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
        return super().forward(**kwargs)
    
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)


class CustomModel(nn.Module):
    def __init__(self,checkpoint,num_labels, seed):
        super(CustomModel,self).__init__()
        
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
            
        self.num_labels = num_labels

        #Load Model with given checkpoint and extract its body
        if USE_LoRA:
            self.model = model = get_peft_model(CustomT5Model.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY)), lora_config)
        else:
            self.model = model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(HIDDEN_SIZE,num_labels) # load and initialize weights
        
    def forward(self, input_ids=None, attention_mask=None,labels=None):
        if 't5-' in checkpoint:
            outputs = self.model(decoder_input_ids=input_ids, input_ids=input_ids, attention_mask=attention_mask)
            encoder_last_hidden_state = outputs.last_hidden_state
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            encoder_last_hidden_state = outputs.last_hidden_state # it is outputs[0]

        #Add custom layers
        sequence_output = self.dropout(encoder_last_hidden_state)

        logits = self.classifier(sequence_output[:,0,:].view(-1,HIDDEN_SIZE)) # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if 't5-' in checkpoint:
            return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions) # orig
        
def generate_file_name():
    file_name = 'Results/' + DATASET
    if FRACTION!=1:
        file_name += '_' + str(FRACTION)
    file_name += '_Input-flan-t5_base_' + INPUT + '_' + checkpoint.split('/')[-1] 
    if ADD_CLASSIFICATION_LAYER:
        file_name += '_wCL'
    else:
        file_name += '_wInf'
    file_name += '_maxlen' + str(MAX_LEN) +  '_bs' + str(BATCH_SIZE) +  '_lr' + str(LR) + '_seed' + str(SEED)
    if USE_LoRA:
        file_name += '_lora'
    return file_name

num_epochs = 8
SEED = 1 # we use 1,2, and 3
ADD_CLASSIFICATION_LAYER = True # replace the last layer with a classification layer (RQ1 and RQ4: True, RQ3: False)
if False: # check other values for hyper parameters
    LR = 0.0001
    BATCH_SIZE = 1


from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW,get_scheduler, AutoModelForSeq2SeqLM
from datasets import load_metric
import torch.nn.functional as F

LABEL_MAX_LEN = 5
def tokenize(batch):
    inputs = tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LEN)
    if ADD_CLASSIFICATION_LAYER:
        return inputs
    else:
        labels = tokenizer([str(label) for label in batch[LABEL_COLUMN]], truncation=True, max_length=LABEL_MAX_LEN, padding='max_length', return_tensors='pt')
#         labels = tokenizer(batch[LABEL_COLUMN], truncation=True, max_length=LABEL_MAX_LEN, padding='max_length', return_tensors='pt')
        inputs["labels"] = labels["input_ids"]
        return inputs
    
metric = load_metric(METRIC) # f1 or accuracy
# metric = load_metric("/root/xinglin-data/satd/f1.py", trust_remote_code=True, local_files_only=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print('Run the experiments for: ' + generate_file_name())


valid_results = {}
valid_losses = {}
test_results = {}
projects_real = {}
projects_pred = {}


for project_name, data in dataset.items():
    print('=======================================')
    print('------', project_name, '------')
    torch.cuda.empty_cache()
    gc.collect()
    valid_results[project_name] = []
    valid_losses[project_name] = []
    test_results[project_name] = []
    projects_real[project_name] = []
    projects_pred[project_name] = []
    labels = [str(x) for x in list(set(dataset[project_name]['train']['label']))]
    
    if ADD_CLASSIFICATION_LAYER:
        model = CustomModel(checkpoint=checkpoint,num_labels=NUM_LABELS, seed=SEED).to(device)
    else:
        if 't5' in checkpoint:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY)).to(device)
        else:
            # model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, output_attentions=True,output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY)).to(device)
            model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, output_attentions=True,output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY)).to(device)
            # model = CustomT5Model.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY)).to(device)
  
    
    # show the model precision
    if False:
        first_param = next(model.parameters())
        print(f"Data type of the first parameter: {first_param.dtype}")
    
    optimizer = AdamW(model.parameters(), lr=LR) 

    tokenized_dataset = data.map(tokenize, batched=True)     
    
    if False:
        # print([len(row['input_ids']) for row in tokenized_dataset['train']])
        print(f'The number of items in the tokenized dataset that their length is {MAX_LEN}. Larger items are also truncated to {MAX_LEN} tokens.')
        print(sum(len(row['input_ids']) == MAX_LEN for row in tokenized_dataset['train']), 'of', len(tokenized_dataset['train']), 'in train dataset')
        print(sum(len(row['input_ids']) == MAX_LEN for row in tokenized_dataset['test']), 'of', len(tokenized_dataset['test']), 'in test dataset')
        raise SystemExit("Stopping the notebook cell execution here.")
    if ADD_CLASSIFICATION_LAYER:
        tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", LABEL_COLUMN])
    else:
        tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "labels"])
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator) # ??? shuffle=True
    test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=data_collator)

    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    i = 0
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            i+=1
            batch = {k: v.to(device) for k, v in batch.items()}            
            if ADD_CLASSIFICATION_LAYER:
                outputs = model(**batch)
                loss = outputs.loss
            else:
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=batch['labels'])
                loss = outputs.loss            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if i%1000==0:
                print(i, 'of', num_training_steps)
        # Free GPU memory
        # print('step',i,'of',num_training_steps)
        print('epoch',epoch+1,'of',num_epochs)
        del batch  # Delete the batch tensor to free GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)

        model.eval()

        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if ADD_CLASSIFICATION_LAYER:
                    outputs = model(**batch)
                    batch_predictions = torch.argmax(outputs.logits, dim=-1)
                    metric.add_batch(predictions=batch_predictions, references=batch["labels"])
                else:
                    # Use the generate method for sequence-to-sequence models
                    generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=LABEL_MAX_LEN)
                    pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
                    label_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["labels"]]
                    pred_texts = ['0' if p not in labels else p for p in pred_texts]
                    metric.add_batch(predictions=pred_texts, references=label_texts)
            # if it is the last epoch, save the predictions
            if epoch==num_epochs-1:
                if ADD_CLASSIFICATION_LAYER:
                    projects_real[project_name] += batch["labels"].tolist()                
                    projects_pred[project_name] += batch_predictions.tolist()
                else:
                    projects_real[project_name] += label_texts                
                    projects_pred[project_name] += pred_texts                    
        test_results[project_name].append(metric.compute())
        print(f"Test:{test_results[project_name][-1][METRIC]:.3f}")
        model.train()

# extract and show f1 score over epochs
results_df = pd.DataFrame({k: [epoch[METRIC] for epoch in v] for k, v in test_results.items()}).T
results_df.columns = ['Epoch'+str(i) for i in range(1,num_epochs+1)]
results_df.loc['Mean'] = results_df.mean()
results_df

# save f1 score to csv file
file_name = generate_file_name()
results_df.to_csv(file_name+'_F1.csv')
print('The results saved in', file_name+'.csv')

# show precision, recall, f1, and confusion matrix
# save the result to [...]_confmat.txt file
def get_confmat_str(real, pred, labels):
    output = ''
    cm = confusion_matrix(real, pred, labels=labels)
    max_label_length = max([len(label) for label in labels] + [5])
    output = " " * max_label_length + " " + " ".join(label.ljust(max_label_length) for label in labels) + "\n"
    for i, label in enumerate(labels):
        row = " ".join([str(cm[i][j]).ljust(max_label_length) for j in range(len(labels))])
        output += label.ljust(max_label_length) + " " + row + "\n"
    return output

if DATASET=='OBrien':
    label_mapping = {
        0: 'Reqmnt', # Requirement
        1: 'Code',
        2: 'M&T',
        3: 'Defect',
        4: 'Design',
        5:'Doc' # Documentation
    }
elif DATASET=='Maldonado62k':
    label_mapping = {
        0: '0',
        1: '1'
    }
else:
    print("ERROR!")

all_real = []
all_pred = []
labels = list(label_mapping.values())

with open(file_name+'_confmat.txt', "w") as output_file:
    for project in projects_pred.keys():
        print('---------- '+str(project)+' ----------')
        output_file.write('\n---------- '+str(project)+' ----------\n')
        real = [label_mapping[int(label)] for label in projects_real[project]]
        pred = [label_mapping[int(label)] for label in projects_pred[project]]
        print(classification_report(real, pred, zero_division=0, digits=3))
        output_file.write(classification_report(real, pred, zero_division=0, digits=3) + '\n')
        # print confusion matrix with label in rows and columns
        confmat_str = get_confmat_str(real, pred, labels=labels)
        print(confmat_str)
        output_file.write(confmat_str)
        # add them to all_real and all_pred
        all_real += real
        all_pred += pred
    print('=========== Overall ==========')
    output_file.write('\n=========== Overall ==========\n')
    # print precision recall and F1 for all data
    print(classification_report(all_real, all_pred, zero_division=0, digits=3))
    output_file.write(classification_report(all_real, all_pred, zero_division=0, digits=3)+"\n")
    # print confusion matrix for all data
    confmat_str = get_confmat_str(all_real, all_pred, labels=labels)
    print(confmat_str)
    output_file.write(confmat_str)


# extract real and pred labels and save to csv
real = []
pred = []
context = []
project = []
for proj in projects_pred.keys():
    for i in range(len(projects_real[proj])):
        project.append(proj)
        context.append(dataset[proj]['test'][TEXT_COLUMN][i])
        real.append(label_mapping[int(projects_real[proj][i])])
        pred.append(label_mapping[int(projects_pred[proj][i])])
test_result_df = pd.DataFrame({'project': project, 'context':context, 'real': real, 'pred': pred})
test_result_df.head(5)
# save to csv file
test_result_df.to_csv(file_name+'_pred.csv', index=False)