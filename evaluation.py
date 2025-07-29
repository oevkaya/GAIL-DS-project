from llmds.params import load_params, input_dataset, data_name_mapping
from llmds.scripts.evaluate import format_namespace
from llmds.scripts.collect_results import main as extract_main
from llmds.scripts.evaluate import main as eval_main


import tiktoken
import json
import os
import time
from types import SimpleNamespace

outcome_path = 'Simulations/output'
outpath = 'Simulations/metrics'
outpath_code = 'Simulations/code_record'

temperature = '1'
model = "gpt-4o"
model_name = model.replace("-","_")
other_setting = True
Qs = [26.0,26.1]

other_sequential = True

datanames = data_name_mapping.keys()
filenames = data_name_mapping.values()
input_metrics='Simulations/metrics/input_metrics.jsonl'

# Load the input of question file
question_input = format_namespace('GAIL-DA-tasks-questions-clean.jsonl')

Qs_dict = {
    'aeroplane': [56,57,58],
    'evals':[21,23,26,23.0,23.1,23.2],
    'weatherAUS':[27,28,29,30,71,30.0,30.1,30.2],
    'UK-visitor-numbers':[i for i in range(10)],
    'instructional-staff': [14.0,14.1,14.2],
    'duke-forest':[59,60,63,65.0,65.1]
}

# data = format_namespace('question.jsonl')
# set_datasets = set([val.file_name for val in data])

# inputs = {}
# for dataset in set_datasets:
#     inputs[dataset] = input_dataset(data,dataset)

# Qs = inputs[filename]['ids'] # only if every question is ready

# ['aeroplane','evals','weatherAUS','UK-visitor-numbers','duke-forest']
for dataname in ['evals']:

    if not other_setting:
        Qs = Qs_dict[dataname]

    args1 = SimpleNamespace(
        dataname = dataname,
        model=model,
        modelname=model_name,
        temperature=temperature,
        # input_folder = f'{outcome_path}/{dataname}',
        input_folder=f'{outcome_path}/{dataname}/Q26_gpt_4o_mini_1.0_code_interpreter_False',
        path=outpath,
        path_code=outpath_code,
        # input=input_metrics,
        question_input=question_input,
        other_setting=other_setting,
        other_sequential=other_sequential,
        Qs=Qs,
        Q_num=len(Qs)
    )

    extract_main(args1)

    args2 = SimpleNamespace(
        input=input_metrics,
        dataname=dataname,
        model=model,
        temperature=temperature,
        other_setting=other_setting,
        # infile=f'{outpath}/{dataname}0_{model_name}_{temperature}.jsonl',
        infile=(f"{outpath}/{dataname}0_{model_name}_{temperature}"
                + ("_no_CodeInterp" if other_setting else "")
                + ".jsonl"),
        path=outpath,
        outfile=(f"{outpath}/{dataname}_{model_name}_{temperature}"
                + ("_no_CodeInterp" if other_setting else "")
                + ".jsonl"),
        other_sequential=other_sequential,
        Qs=Qs,
        Q_num=len(Qs)
    )

    eval_main(args2)