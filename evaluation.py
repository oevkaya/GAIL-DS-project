from llmds.params import load_params, input_dataset, data_name_mapping
from llmds.scripts.evaluate import format_namespace
from llmds.scripts.collect_results import main as extract_main
from llmds.scripts.evaluate import main as eval_main


import tiktoken
import json
import os
import time
from types import SimpleNamespace

temperature = '1'
model = "gpt-4o"
model_name = model.replace("-","_")
outpath = "Simulations/metrics"

datanames = data_name_mapping.keys()
filenames = data_name_mapping.values()
input_metrics='Simulations/metrics/input_metrics.jsonl'

# Load the input of question file
question_input = format_namespace('GAIL-DA-tasks-questions-clean.jsonl')

Qs = [27,28,29,30,72]
# data = format_namespace('question.jsonl')
# set_datasets = set([val.file_name for val in data])

# inputs = {}
# for dataset in set_datasets:
#     inputs[dataset] = input_dataset(data,dataset)

# Qs = inputs[filename]['ids'] # only if every question is ready

for dataname in ['weatherAUS']:

    args1 = SimpleNamespace(
        dataname = dataname,
        model=model,
        modelname=model_name,
        temperature=temperature,
        input_folder = f'Simulations/output/{dataname}',
        path='Simulations/metrics',
        input=input,
        question_input=question_input
    )

    extract_main(args1)

    args2 = SimpleNamespace(
        input=input_metrics,
        model=model,
        infile=f'Simulations/metrics/{dataname}0_{model_name}_{temperature}.jsonl',
        dataname=dataname,
        outfile=f'Simulations/metrics/{dataname}_{model_name}_{temperature}.jsonl',
        Qs=Qs,
        Q_num=len(Qs)
    )

    eval_main(args2)