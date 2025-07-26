from llmds.params import format_namespace, data_name_mapping, call_question_by_ids, fileid_name_mapping
from llmds.params import csv_to_text
# from llmds.dm import input_dataset
# from llmds.scripts.evaluate import format_namespace
from llmds.scripts.automation import multi_round_assistant, sequential_question_assistant

import json
import os
import time
from openai import OpenAI
import argparse
from types import SimpleNamespace


# inputs = {}
# for dataset in set_datasets:

#     inputs[dataset] = input_dataset(data,dataset)

# dinput = inputs[filename]

# Generate the metrics from the input set of DS questions. 
#compute_input_metrics(data,"Simulations/metrics/input.jsonl",model_choice)

openai_key="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"

client=OpenAI(
    api_key=openai_key
)

def main(args):
  for dataname in args.datanames:
    for model in args.models:
      for temperature in args.temperatures:

        filename = data_name_mapping[dataname]
        file_id = fileid_name_mapping[dataname]
        outfolder = f'Simulations/output/{dataname}/Q{args.Qs[0]}_{model.replace("-","_")}_{temperature}'
        if not os.path.exists(outfolder):
          os.makedirs(outfolder, exist_ok=True)

        content_set = call_question_by_ids(args.data,args.Qs)
        if args.with_code_interpreter: 
          #-----------create an assistant with code interpreter---------------
          assistant = client.beta.assistants.create (
            name = "Question and Code Assistant",
            instructions =args.instruction,
            model = model,
            tools = [{'type': 'code_interpreter'}],
            temperature=temperature,
            tool_resources={ 
              'code_interpreter': {
              'file_ids': [file_id]
              }
          })
        else:
          #-----------create an assistant without code interpreter---------------
          assistant = client.beta.assistants.create (
            name = "Question and Code Assistant",
            instructions =args.instruction,
            model = model,
            tools = [{'type': 'file_search'}],
            temperature=temperature,
            tool_resources={
            'file_search': {"vector_store_ids": [args.vector_store_id]}}
          )

        assistant_id = assistant.id
        if not args.sequential:# If questions are independent:
          assistant_id = multi_round_assistant(client,assistant_id,args.Qs,content_set,outfolder,args.ks,args.K) 
        else: # If questions are sequential:
          assistant_id = sequential_question_assistant(client,assistant_id,args.Qs,content_set,outfolder,args.ks,args.K)

        client.beta.assistants.delete(assistant_id)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run experiment groups with assistants.")
  
  args = SimpleNamespace(
    data=format_namespace('GAIL-DA-tasks-questions-clean.jsonl'),
    instruction="""You are a specialized assistant for iterative data‐science tasks. Every time the user asks a question or provides data.
  Answer each question. You will return a JSON object with one key: `"outcome"`, which is a string or JSON array describing the results.
  """,
    datanames=['cherryblossom'],
    models=["gpt-4.1-nano"],
    temperatures=[1.0],
    Qs=[34],
    sequential=False,
    with_code_interpreter=True,
    vector_store_id='vs_6881c6c49494819185527bbb078f62b3',
    ks=[0], # if non-sequential
    # ks=0 # if sequential
    K=100
  )

  main(args)