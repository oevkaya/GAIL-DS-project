import tiktoken
import json
import numpy as np
import argparse
import os
from types import SimpleNamespace
# from collections import defaultdict

def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def compute_input_metrics(data,infile,model):

    with open(infile, 'w') as f:
        for item in data:
            q = item.question

            input_metric = {
                'id': item.id,
                'input_words':len(q),
                'input_tokens': count_tokens(q,model)
            }

            f.write(json.dumps(input_metric) + "\n")


def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]
    
def format_namespace(path):
   with open(path, 'r') as f:
      return [SimpleNamespace(**json.loads(line)) for line in f]

def combine_into_one_string(text):
    return " ".join(text)

def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

def compute_similarity(data,Q,method):
   id_to_reasoning = [item.reasoning for item in data if item.id == Q]
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

def main(args):
    print(f"Evaluation for {args.dataname}")

    inmetrics = format_namespace(args.input)
    outmetrics = format_namespace(args.infile)

    #-----add completion ratio (CR) and verbosity ratios   
    status_mapping = {
        'completed':1,
        'failed':0
        }
    
    id_map = {item.id: item for item in inmetrics}

    for item in outmetrics:
        if item.id in id_map:
            for para, val in vars(id_map[item.id]).items():
                if para != 'id':
                    setattr(item, para, val)

    for item in outmetrics:
        item.verbosity_ratio_words = item.input_words / item.words if item.words !=0 else None 
        item.verbosity_ratio_tokens = item.input_tokens / item.tokens if item.tokens != 0 else None 

        item.CR = status_mapping[item.status] if item.status in status_mapping else None 

    #-----add text similarity
    id_jaccard_values = {}

    for Q in range(args.Q_num):
        id_jaccard_values[args.Qs[Q]] = compute_similarity(outmetrics, args.Qs[Q], 'Jaccard')


    for item in outmetrics:
       id_, round_ = item.id, item.round 

       if id_ in id_jaccard_values: #and round_ in id_jaccard_values[id_]:
          item.Similarity_Jaccard = id_jaccard_values[id_][round_]
       else:
          item.Similarity_Jaccard = None 

    with open(args.outfile, "w", encoding="utf-8") as f:
        for item in outmetrics:
           f.write(json.dumps(vars(item), ensure_ascii=False)+"\n")

    os.remove(args.infile)

if __name__ == '__main__':
   parser = argparse.ArgumentParser()

   parser.add_argument('--input', type=str, default='Simulations/metrics/input.jsonl')
   parser.add_argument('--infile', type=str, default='Simulations/metrics/weatherAUS0.jsonl')
   parser.add_argument('--dataname',type=str,default='weatherAUS')
   parser.add_argument('--outfile', type=str, default='Simulations/metrics/weatherAUS.jsonl')

   parser.add_argument('--Q_num', type=int)
   parser.add_argument('--Qs',type=list)

   args = parser.parse_args()
   main(args)
