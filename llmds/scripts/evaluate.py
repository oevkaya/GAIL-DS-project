import tiktoken
import json
import numpy as np
import argparse
import os
from types import SimpleNamespace
from llmds.params import format_namespace
# from collections import defaultdict

def count_tokens(text, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

# def count_tokens(text, model):
    #   """Try this if the above one doesn't work """
#     if model in ["gpt-4.1-mini","gpt-4.1-nano"]:
#         encoding = tiktoken.get_encoding("cl100k_base")
#     else:
#         encoding = tiktoken.encoding_for_model(model)

#     return len(encoding.encode(text))

def compute_input_metrics(data,infile,models):

    with open(infile, 'w') as f:
        for item in data:
            q = item.question

            input_metric = {
                'id': item.id,
                'input_words': len(q)
            }

            for model in models:
                label = f"input_tokens_{model.replace('-', '_')}"
                input_metric[label] = count_tokens(q,model)

            f.write(json.dumps(input_metric) + "\n")


def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]
    
# def format_namespace(path):
#    with open(path, 'r') as f:
#       return [SimpleNamespace(**json.loads(line)) for line in f]

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

def find_lines_with_keyword(text, keyword):
    return [line for line in text.splitlines() if keyword.lower() in line.lower()]

def line_contains_feature(line, feature):
    # If feature is string or number, do simple substring match
    if isinstance(feature, (str, int, float)):
        return str(feature).lower() in line.lower()

    # If feature is a dict, attempt to parse line as JSON and compare structure
    elif isinstance(feature, dict):
        try:
            obj = json.loads(line)
            return dict_contains(obj, feature)
        except Exception:
            return False
    return False

def dict_contains(container, sub):
    """
    Recursively check if 'sub' dict structure and values are in 'container'.
    """
    if not isinstance(container, dict) or not isinstance(sub, dict):
        return False
    for key, val in sub.items():
        if key not in container:
            return False
        if isinstance(val, dict):
            if not dict_contains(container[key], val):
                return False
        else:
            if container[key] != val:
                return False
    return True

def find_lines_with_features(text, features, tail_lines=3):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    tail = lines[-tail_lines:] 

    matches = []
    for line in tail:
        if any(line_contains_feature(line, f) for f in features):
            matches.append(line)
    return matches


def delete_empty_folders(folder_path):
    for foldername, subfolders, filenames in os.walk(folder_path, topdown=False):
        if not os.listdir(foldername):  # Folder is empty
            os.rmdir(foldername)
            print(f"Deleted empty folder: {foldername}")

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
        item.verbosity_ratio_tokens = getattr(item, f"input_tokens_{args.model.replace('-', '_')}") / item.tokens if item.tokens != 0 else None 

        item.complete_ratio = status_mapping[item.status] if item.status in status_mapping else None 
        
        if hasattr(item, 'common_answers'): 
           item.accuracy = 1 if find_lines_with_features(item.reasoning,item.common_answers) else 0

        # item.code_execute 
    #-----add text similarity
    # id_jaccard_values = {}

    # for Q in range(args.Q_num):
    #     id_jaccard_values[args.Qs[Q]] = compute_similarity(outmetrics, args.Qs[Q], 'Jaccard')


    # for item in outmetrics:
    #    id_, round_ = item.id, item.round 

    #    if id_ in id_jaccard_values: #and round_ in id_jaccard_values[id_]:
    #       item.Similarity_Jaccard = id_jaccard_values[id_][round_]
    #    else:
    #       item.Similarity_Jaccard = None 

    # with open(args.outfile, "w", encoding="utf-8") as f:
    #     for item in outmetrics:
    #        f.write(json.dumps(vars(item), ensure_ascii=False)+"\n")

    os.remove(args.infile)

if __name__ == '__main__':
   parser = argparse.ArgumentParser()

   parser.add_argument('--input', type=str, help='File of input metrics')
   parser.add_argument('--infile', type=str, help='Initial version of metrics data, obtained from component extraction.')
   parser.add_argument('--dataname',type=str, help='Dataset name')
   parser.add_argument('--outfile', type=str, help='Final version of metrics')

   parser.add_argument('--Q_num', type=int)
   parser.add_argument('--Qs',type=list)

   args = parser.parse_args()
   main(args)
