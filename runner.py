from llmds.params import load_params, input_dataset, data_name_mapping
# from llmds.dm import input_dataset
from llmds.scripts.evaluation import format_namespace
from llmds.scripts.automation import multi_round_assistant, sequential_question_assistant

import tiktoken
import json
import os
import time
from openai import OpenAI

# Required parameters
model_choice = "gpt-4o"
tem = 1.0
instruction = """
  You are a specialized assistant for iterative data‐science tasks. Every time the user asks a question or provides data.
  Answer each question.
  You will return a JSON object with one key: `"outcome"`, which is a string or JSON array describing the results.
  """

dataname = 'instructional-staff'
filename = data_name_mapping[dataname]
file_id = 'file-5riwCDAHXme7U6mfNZwCLe'

# data = load_params("GAIL-DA-tasks-questions.jsonl")
data = format_namespace('GAIL-DA-tasks-questions-clean.jsonl')

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
outfolder = f'Simulations/output/{dataname}'

# Setup for multiple questions specific to one dataset
# Qs = [34,35]
# Ns = [91,80]
# ks = [100 - n for n in Ns]

# Setup for sequential questions specific to one dataset
Q_id = 14
Qs = [14,14.1,14.2]
N = 30
ks = 20


#---------------create an assistant---------------
assistant = client.beta.assistants.create (
  name = "Question and Code Assistant",
  instructions =instruction,
  model = model_choice,
  tools = [{'type': 'code_interpreter'}],
  temperature=tem,
  tool_resources={ 
    'code_interpreter': {
      'file_ids': [file_id]
      }
  })

assistant_id = assistant.id


# Run for different temperatures:
# for tem in [0.5,1,5]:
#     assistant_id = multi_round_assistant(client,Qs,dinput,model_choice,file_id,Ns,outfolder,ks,tem)   

# assistant_id = multi_round_assistant(client,assistant_id,Qs,dinput,Ns,outfolder,ks) 

# assistant_id = multi_round_assistant(client,Qs,dinput,"gpt-4o-mini",file_id,Ns,f'Simulations/output/{dataname}/others3',ks,1.0) 

assistant_id = sequential_question_assistant(client,assistant_id,Q_id,Qs,dinput,N,outfolder,ks)

client.beta.assistants.delete(assistant_id)