# from llmds.scripts.evaluation import count_tokens, compute_input_metrics
from llmds.scripts.collect_results import main as extract_main
from llmds.scripts.evaluation import main as eval_main

import tiktoken
import json
import os
import time
from openai import OpenAI
from collections import defaultdict
from types import SimpleNamespace

model_choice = "gpt-4o"
dataname = 'weatherAUS'

outfolder = f'Simulations/output/{dataname}'
outpath = "Simulations/metrics"

args1 = SimpleNamespace(
    input_folder='Simulations/output/weatherAUS',
    path='Simulations/metrics',
    dataname='weatherAUS',
    model='gpt-4o'
)

args2 = SimpleNamespace(
    input='Simulations/metrics/input.jsonl',
    infile='Simulations/metrics/weatherAUS0.jsonl',
    dataname='weatherAUS',
    outfile='Simulations/metrics/weatherAUS.jsonl',
    Q_num=3,
    Qs=[27,28,29]
)

extract_main(args1)
eval_main(args2)

