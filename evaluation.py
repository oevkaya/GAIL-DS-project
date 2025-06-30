from llmds.params import load_params, input_dataset, data_name_mapping
from llmds.scripts.evaluation import format_namespace
from llmds.scripts.collect_results import main as extract_main
from llmds.scripts.evaluation import main as eval_main


import tiktoken
import json
import os
import time
from types import SimpleNamespace

model_choice = "gpt-4o"
dataname = 'laptop_data_cleaned'
# filename = 'laptop_data_cleaned.csv'
filename = data_name_mapping[dataname]
model_name = 'gpt_4o'
temperature = '1'

outfolder = f'Simulations/output/{dataname}'
outpath = "Simulations/metrics"

data = format_namespace('questions.jsonl')
set_datasets = set([val.file_name for val in data])

inputs = {}
for dataset in set_datasets:
    inputs[dataset] = input_dataset(data,dataset)

# Qs = inputs[filename]['ids']
Qs = [49,50]
Q_num = len(Qs)

args1 = SimpleNamespace(
    input_folder=outfolder,
    path='Simulations/metrics',
    dataname=dataname,
    model=model_choice,
    modelname=model_name,
    temperature=temperature
)

args2 = SimpleNamespace(
    input='Simulations/metrics/input.jsonl',
    infile=f'Simulations/metrics/{dataname}0_{model_name}_{temperature}.jsonl',
    dataname=dataname,
    outfile=f'Simulations/metrics/{dataname}_{model_name}_{temperature}.jsonl',
    Q_num=Q_num,
    Qs=Qs
)

extract_main(args1)
eval_main(args2)

