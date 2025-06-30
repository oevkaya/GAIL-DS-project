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
