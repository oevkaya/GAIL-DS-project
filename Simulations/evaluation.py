import json
import os
import re
import numpy as np
import tiktoken


# data from the GAIL-DA-tasks-questions.jsonl

#---------------Count the input words/tokens---------------
# count the words and tokens of input
model_choice = "gpt-4o"
def count_tokens(text, model=model_choice):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

input_file = "Simulations/metrics/input.jsonl"
with open(input_file, 'w') as f:
    for item in data:
        q = item.get('question')

        input_metric = {
            'id': item.get('id'),
            'input_words':len(q),
            'input_tokens': count_tokens(q)
        }

        f.write(json.dumps(input_metric) + "\n")


#---------------Combine input and output metrics---------------
def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]
    
status_mapping = {
    'completed':1,
    'failed':0
}

input_file = "Simulations/metrics/input.jsonl"
response = "Simulations/metrics/UK-visitor-numbers.jsonl"
output_file = "Simulations/metrics/visitor2.jsonl"

input_metrics = read_jsonl(input_file)
output_metrics = read_jsonl(response)

id_to_info = {entry['id']: entry for entry in input_metrics}

input_para = ['input_words','input_tokens']
# metrics_to_keep = {key for key in data}
# metrics_to_keep.difference_update({'outcome','reasoning','thread_id'}) 

with open(output_file,'w') as outfile:

    for data in output_metrics:
        id = data.get('id')

        if id in id_to_info:
            data.update({k:id_to_info[id].get(k) for k in input_para})
            data.update()

        input_words = data.get('input_words')
        input_tokens = data.get('input_tokens')
        words = data.get('words')
        tokens = data.get('tokens')

        data['verbosity_ratio_words'] = input_words / words
        data['verbosity_ratio_tokens'] = input_tokens / tokens
        status = data.get('status')
        if status in status_mapping:
            data['CR'] = status_mapping[status]

        # filtered_data = []

        outfile.write(json.dumps(data) + "\n")

#---------------Add text similarity---------------
# Text similarity: Jaccard Index, Euclidean Distance, Cosine Similarity

# from math import sqrt, pow, exp

def combine_into_one_string(text):
    return " ".join(text)

def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

def compute_similarity(data,Q,method):
   id_to_reasoning = [entry.get('reasoning') for entry in data if entry.get('id') == Q]
   sentences = [combine_into_one_string(sentence) for sentence in id_to_reasoning]
   num = len(sentences)

   if method == 'Jaccard':
      base = sentences[0]
      values = np.zeros(num)
      for i in range(num):
         values[i] = jaccard_similarity(base,sentences[i])
      
      return values
    
   elif method == 'Euclidean':
    pass
    #embeddings = [nlp(sentence).vector for sentence in sentences]
   elif method == 'Cosine':
    pass
      
metrics = read_jsonl(output_file)
id_jaccard_values = {}
for Q in range(9):
    id_jaccard_values[Q] = compute_similarity(metrics, Q, 'Jaccard')

path = "Simulations/metrics"
output_file1 = f"{path}/visitor2.jsonl"
output_file = f"{path}/UK-visitor-numbers.jsonl"
with open(output_file1, 'r') as f, open(output_file,'w') as outf:
    for line in f:
        data = json.loads(line)

        id_ = data.get('id')
        round_ = data.get('round')

        if id_ in id_jaccard_values:
            data['Similarity_Jaccard'] = id_jaccard_values[id_][round_]

        outf.write(json.dumps(data) + "\n")


#--------------Old version: Used for Evaluations--------------

def evaluate_per_question(contents):
  """ 
  Evaluation metrics include 
  - average runtime 
  - completion ratio
  - average number of words / tokens

  added soon
  """
  
  num_of_rounds = len(contents)

  runtimes = [d['runtime'] for d in contents]
  average_runtime = sum(runtimes) / num_of_rounds

  completions = [1 for d in contents if d['status'] == 'completed']

  completion_ratio = sum(completions) / num_of_rounds

  words = [d['words'] for d in contents]
  average_words = sum(words)/num_of_rounds

  tokens = [d['tokens'] for d in contents]
  average_tokens = sum(tokens)/num_of_rounds

  return average_runtime, completion_ratio, average_words, average_tokens
  # return runtimes, words, tokens, completions


Q_list = [1]
evaluation_list = {}

for Q in Q_list:
  responses = f"Simulations/responses/Q{Q}.jsonl"
  contents = []
  with open(responses, "r") as f:
    for line in f:
      content = json.loads(line)
      contents.append(content)

  evaluation_list[Q] = evaluate_per_question(contents)

print(evaluation_list)

