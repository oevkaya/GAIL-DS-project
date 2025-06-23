from llmds.params import load_params, dict_datasets, input_dataset
# from llmds.dm import input_dataset
from llmds.scripts.automation import multi_round_assistant

import tiktoken
import json
import os
import time
from openai import OpenAI

# Required parameters
model_choice = "gpt-4o"
dataname = 'UK-visitor-numbers'
filename = 'UK-visitor-numbers.csv'
file_id = 'file-LB38GMvxj7SfRMPrenaBR1'


data = load_params("GAIL-DA-tasks-questions.jsonl")

set_datasets = set([val.file_name for val in data])

inputs = {}
for dataset in set_datasets:

    inputs[dataset] = input_dataset(data,dataset)

# Generate the metrics from the input set of DS questions. 
#compute_input_metrics(data,"Simulations/metrics/input.jsonl",model_choice)

#Run multi-round experiments
openai_key="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"

client=OpenAI(
    api_key=openai_key
)

dinput = inputs[filename]
Qs = [1,2,3,4,6,7,8,9]
Ns = [39,50,50,50,50,64,50,50]
ks = [100 - n for n in Ns]

outfolder = f'Simulations/output/{dataname}'

assistant_id = multi_round_assistant(client,Qs,dinput,model_choice,file_id,Ns,outfolder,ks)   

client.beta.assistants.delete(assistant_id)