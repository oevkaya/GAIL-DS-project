#--------------Used for Evaluations--------------
import json

Q = 1
responses = f"Simulations/responses/Q{Q}.jsonl"
contents = []
with open(responses, "r") as f:
  for line in f:
     content = json.loads(line)
     contents.append(content)


def evaluate(contents):
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

evaluate(contents)