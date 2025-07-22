import pickle
import argparse
import re
import tiktoken
import json
import os
import glob
import numpy as np
from types import SimpleNamespace

from llmds.params import format_namespace
#------------------Extraction------------------
def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def extract_code_blocks(text):
    pattern = r"```(?:python)?\n(.*?)```"
    # pattern = r"```(\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)

# def extract_reasoning(text):
#   """ 
#   Old version: Remove Python/R/other language code and the header from text 
#   """
#   pattern = r"```(?:python)?\n(.*?)```"
#   # pattern = r"```(\w+)?\n(.*?)```"
#   header = "Role: assistant"

#   def clean_block(text):
#      text = re.sub(pattern, '', text, flags=re.DOTALL)
#      text = re.sub(header, '', text).strip()

#      return text

#   if isinstance(text, list):
#     return [clean_block(t) for t in text]
#   else:
#     return clean_block(text)
  
def clean_block(pattern, text):
  text = re.sub(pattern, '', text, flags=re.DOTALL)

  return text

# def extract_reasoning(text):
#   """ 
#   Remove Python/R/other language code and the header from text 
#   """
#   pattern = r"```(?:python)?\n(.*?)```"
#   pattern2 = r'^Role: assistant\s*'
#   # def clean_block(pattern, text):
#   #    text = re.sub(pattern, '', text, flags=re.DOTALL)

#   #    return text

#   if isinstance(text, list):
#     return [clean_block(pattern,t) for t in text]
#   else:
#     return clean_block(pattern,text)

def extract_reasoning(text):
  """ 
  Remove Python/R/other language code, the header from text, and newline characters
  """
  text = re.sub(r"```(?:python)?\n(.*?)```",'',text,flags=re.DOTALL)
  text = re.sub(r'^Role: assistant\s*','',text,flags=re.IGNORECASE)
  text = text.replace('\n', '')

  return text

def extract_metadata(header_block, image=False):
    """Extract round ID, thread ID, runtime, status, image_id from the first block."""
    round_match = re.search(r"Round\s+(\d+)", header_block)
    thread_match = re.search(r"thread_id:\s*(\S+)", header_block)
    runtime_match = re.search(r"Runtime:\s*([\d.]+)", header_block)
    status_match = re.search(r"Status:\s*(\w+)", header_block)
    
    image_match = None
    if image:
        image_match = re.search(r"image_id:\s*(\S+)", header_block)

    return {
        "round": int(round_match.group(1)) if round_match else None,
        "thread_id": thread_match.group(1) if thread_match else None,
        "runtime": float(runtime_match.group(1)) if runtime_match else None,
        "status": status_match.group(1) if status_match else None,
        "image_id": image_match.group(1) if image_match else None
    }

def split_by_user(messages):
    """Used for sequential questions. 
    Separate the outcomes into per question.
    """
    conversations = []
    current = []

    for msg in messages:
        if msg.startswith('Role: user'):
            if current:
                conversations.append(current)
            current = [msg]  # start new block
        else:
            current.append(msg)

    if current:
        conversations.append(current)  # add the last block

    return conversations

# def extract_metadata(header_block):
#     """Extract round ID, thread ID, runtime, and status from the first block."""
#     round_match = re.search(r"Round\s+(\d+)", header_block)
#     thread_match = re.search(r"thread_id:\s*(\S+)", header_block)
#     runtime_match = re.search(r"Runtime:\s*([\d.]+)", header_block)
#     status_match = re.search(r"Status:\s*(\w+)", header_block)

#     return {
#         "round": int(round_match.group(1)) if round_match else None,
#         "thread_id": thread_match.group(1) if thread_match else None,
#         "runtime": float(runtime_match.group(1)) if runtime_match else None,
#         "status": status_match.group(1) if status_match else None
#     }


# def extract_outcome(text):
#     """
#     old version: Extracts phrases like 'contains 348', 'are 25', or 'has 109' from assistant responses.
#     Returns the first match or None.
#     """
#     pattern = r"(?:contains|are|has|include|comprise|involve)\s+(about\s+)?(\d+[,\d]*)(\s+\w+)?"
#     match = re.search(pattern, text, re.IGNORECASE)
#     return match.group(0).strip() if match else None


# def extract_outcome(text):
#     """
#     Ways to capture outcomes:
#     1. Extracts phrases like 'contains 348', 'are 25', or 'has 109' from assistant responses.
#     Returns the first match or None.
#     2. Extract the content from the last assistant except the code, find any json format.
#     """

#     pattern = r"(?:contains|are|has|include|comprise|involve)\s+(about\s+)?(\d+[,\d]*)(\s+\w+)?"
#     match = re.search(pattern, text, re.IGNORECASE)
#     num_part = match.group(0).strip() if match else None

    
#     pattern2 = r"```json\s*(\{.*?\})\s*```"
#     match2 = re.search(pattern2, text, re.IGNORECASE)
#     # json_part = match2.group(0).strip() if match2 else None

#     if num_part:
#         return num_part
#     elif match2:
#         json_part = match2.group(1)
#         try:
#             out = json.loads(json_part)
#             return out.get('outcome')
#             # return json_part
#         except json.JSONDecodeError:
#             pass
#     else:
#       pass
      #return clean_block(r"```(?:python)?\n(.*?)```",text)

# def extract_code_for_evaluation(Q,section,round_num,thread_id,codefile=None):
#   code = extract_code_blocks(section)
#   code_part = code[0] if code else None
  
#   if not os.path.exists(codefile):
#     with open(codefile, "w") as f:
#        pass

#   # Save code block
#   if code_part:
#     with open(codefile, "a", encoding="utf-8") as f:
#       f.write("#"*50)
#       f.write(f"\n#Question {Q}, Round {round_num} with threat_id: {thread_id}\n")
#       f.write(code_part)


# def extract_response_for_evaluation(Q, section,model,jsonfile=None):

#     # Split responses by 50-dash separator
#     blocks = [block.strip() for block in section.split("-" * 50) if block.strip()]

#     elements = extract_metadata(blocks[0])
#     outcome = extract_outcome(blocks[-1])
#     reasoning = []

#     content_token = 0
#     content_word = 0

#     for block in blocks[1:]:
#       # Skip user blocks
#       if block.startswith("Role: user"):
#         continue

#       if block.startswith("Role: assistant"):
#         #   block_clean = clean_block(block)
#           content_token += count_tokens(block,model)
#           content_word += len(block.split())

#           reasoning.append(extract_reasoning(block))
    
#     # outcome = extract_outcome(reasoning[-1])
#     #--------------Save each component--------------
#     if not os.path.exists(jsonfile):
#       with open(jsonfile, "w") as f:
#         pass
    
#     out_block = SimpleNamespace(
#        id=Q, 
#        round=elements['round'],
#        thread_id=elements['thread_id'],
#        status=elements['status'],
#        runtime=elements['runtime'],
#        words=content_word,
#        tokens=content_token,
#        reasoning=reasoning,
#        outcome=outcome
#     )
#     with open(jsonfile, "a", encoding="utf-8") as f:
#        f.write(json.dumps(out_block.__dict__)+"\n")
       
#     # json_block = {
#     #    "id":Q, 
#     #    "round":elements['round'],
#     #    "thread_id":elements['thread_id'],
#     #    "status":elements['status'],
#     #    "runtime":elements['runtime'],
#     #    "words":content_word,
#     #    "tokens":content_token,
#     #    "reasoning": reasoning,
#     #    "outcome": outcome
#     # }   
    
#     # with open(jsonfile, "a", encoding="utf-8") as f:
#     #   json.dump(json_block, f, ensure_ascii=False)
#     #   f.write("\n")

# def main(args):
#   print(f"Extract components for {args.dataname}")
    
#   # For sequential questions
#   infiles = glob.glob(os.path.join(args.input_folder, '*.txt'))
#   infiles = [f for f in infiles if '_multi' not in os.path.basename(f)]

#   outfile = f"{args.path}/{args.dataname}0_{args.modelname}_{args.temperature}.jsonl"
#   codefile = f"{args.path}/{args.dataname}_{args.modelname}_{args.temperature}.py"

#   for text_file in infiles:
#     print(text_file)

#   #   match = re.match(r"Q(\d+(?:\.\d+)?)\.txt", os.path.basename(text_file))
#     Q = find_number(text_file)

#     with open(text_file, 'r') as f:
#       content = f.read()

#       # Find the section for each thread_id, separated by 100-dash separator
#       sections = [section.strip() for section in content.split("-" * 100) if section.strip()]

#       for section in sections:
#         id_match = re.search(r"Round (\d+)\s+with thread_id:\s+(\w+)", section)
#         if id_match:
#           round_num = int(id_match.group(1))
#           thread_id = id_match.group(2)

#         # Choose the version of json/jsonl, or separate code/reasoning/outcomes files. 
#         extract_response_for_evaluation(Q, section,args.model,jsonfile=outfile)

#         extract_code_for_evaluation(Q,section,round_num,thread_id,codefile=codefile)

def find_question_basics(folder_path):
    """For each file within folder_path, identify the question id,
    filename, whether sequential, and whether with image outcomes
    """
    # Regex to match text files and image folders
    txt_pattern = re.compile(r'^Q(\d+)(_multi)?\.txt$')
    folder_pattern = re.compile(r'^Q(\d+)_image$')

    # First, collect all folder IDs with _image
    image_folders = set()
    for name in os.listdir(folder_path):
        match = folder_pattern.match(name)
        full_path = os.path.join(folder_path, name)
        if match and os.path.isdir(full_path):
            q_id = int(match.group(1))
            image_folders.add(q_id)

    # Now collect question file entries (allowing duplicates)
    questions = []

    for filename in os.listdir(folder_path):
        match = txt_pattern.match(filename)
        if match:
            q_id = int(match.group(1))
            is_multi = match.group(2) is not None
            has_image = q_id in image_folders
            questions.append(SimpleNamespace(id=q_id, is_multi=is_multi, image_exist=has_image, filename=filename))

    return questions

def separate_text_and_code(Q,section,round_num,thread_id,model,codefile=None):
    # Extract code and keep text while cleaning the headers, i.e. Role: assistant
    reasoning_part, code_part = [], []

    content_token, content_word = 0, 0

    for block in section:
        if block.startswith('Role: user'):
            continue

        if block.startswith('Role: assistant'):
            
            content_token += count_tokens(block,model)
            content_word += len(block.split())

            reasoning_part.append(extract_reasoning(block))

            code = extract_code_blocks(block)
            if code:
                code_part.append(code[0])

    out_block = SimpleNamespace(
       id=Q,
       round=round_num,
       thread_id=thread_id,
       words=content_word,
       tokens=content_token,
       reasoning=' '.join(reasoning_part),
    )
        
    #--------------Save each component--------------
    if not os.path.exists(codefile):
        with open(codefile, "w") as f:
            pass
    
    if code_part:
        # if len(code_part) > 1:
        code_part = "\n\n".join(code_part)

        #     print(f'Round {round_num}: code shown more than once')
        with open(codefile, "a", encoding="utf-8") as f:
            f.write("#"*50)
            f.write(f"\n#Question {Q}, Round {round_num} with threat_id: {thread_id}\n")
            f.write(code_part)

    return out_block


def find_number(text_file):
    match = re.match(r"Q(\d+(?:\.\d+)?)\.txt", os.path.basename(text_file))
    x = float(match.group(1))
    return int(x) if x == int(x) else str(x)

def main(args):
  print(f"Extract components for {args.dataname}")
  
  # Find all the files and identify sequential questions stored in Q.._multi.txt
  infiles = glob.glob(os.path.join(args.input_folder, '*.txt'))
  # sequential_infiles = [f for f in infiles if '_multi' in os.path.basename(f)]
  # sole_infiles = [f for f in infiles if '_multi' not in os.path.basename(f)]

  outfile = f"{args.path}/{args.dataname}0_{args.modelname}_{args.temperature}.jsonl"
  codefile = f"{args.path}/{args.dataname}_{args.modelname}_{args.temperature}.py"

  # check whether questions require image outputs
  # image_folders = glob.glob(os.path.join(args.input_folder, '*_image'))
  
  # Load the input of question file
  # question_input = format_namespace('GAIL-DA-tasks-questions-clean.jsonl')
  questions = find_question_basics(args.input_folder)

  for text_file in infiles:
    print(text_file)

    q_type = list(filter(lambda item: item.filename == os.path.basename(text_file), questions))[0]
    Q, sequential, image_exist = q_type.id, q_type.is_multi, q_type.image_exist

    q_input = list(filter(lambda item: int(item.id) == Q, args.question_input))
    q_ids = [item.id for item in q_input]

    if sequential: 
      q_ids[0] = float(q_ids[0])

    with open(text_file, 'r') as f:
      content = f.read()

      # Split into answer per round
      sections = [section.strip() for section in content.split("-" * 100) if section.strip()]

      for section in sections:
        # Split into answer per assistant role 
        blocks = [block.strip() for block in section.split("-" * 50) if block.strip()]

        # Extract the basic information from the header block
        elements = extract_metadata(blocks[0], image_exist)
        
        lines = []
        if sequential:# add the summary header
          header = SimpleNamespace(
            id = f"{int(q_ids[0])}_multi",
            round=elements['round'],
            thread_id=elements['thread_id'],
            status=elements['status'],
            runtime=elements['runtime'],
            words=0,
            tokens=0,
            reasoning=None
          )

          lines.append(header)
          # with open(outfile, "a", encoding="utf-8") as f:
          #   f.write(json.dumps(header.__dict__)+"\n")
  

        # Separate by question 
        subblocks = split_by_user(blocks[1:])

        basic_to_add = {'status':elements['status'],'runtime':0}

        for i, subblock in enumerate(subblocks):
          txt_block = separate_text_and_code(q_ids[i],subblock,elements['round'],elements['thread_id'],args.model,codefile)

          if not sequential:
            basic_to_add = {'status':elements['status'],'runtime':elements['runtime']}

          for key, val in basic_to_add.items():
            setattr(txt_block, key, val)
          
          lines.append(txt_block)

        if sequential:
          # One further step is to assign the runtime to each question based on the ratio of words

          header_record = next(r for r in lines if str(r.id).endswith("_multi"))
          sub_records = [r for r in lines if not str(r.id).endswith("_multi")]

          total_words = sum(r.words for r in sub_records)
          
          for r in sub_records:
            proportion = r.words / total_words if total_words else 0
            r.runtime = round(header_record.runtime * proportion, 3)

          lines = sub_records

        # Write to a jsonl file
        with open(outfile, "a", encoding="utf-8") as f:
          for line in lines:
            f.write(json.dumps(line.__dict__)+"\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--input_folder', type=str, help='Folder path of the collection of outputs for the dataset.')
  parser.add_argument('--path', type=str, help='Path to store the metrics.',default='Simulations/metrics')
  parser.add_argument('--dataname',type=str,help='Dataset name')
  parser.add_argument('--model',type=str,help='Model choice, e.g. gpt-4o')
  parser.add_argument('--modelname',type=str,help='Model, e.g. gpt_4o')
  parser.add_argument('--temperature',type=str,help='Temperature, e.g. 1.0')

  parser.add_argument('--question_input', type=str,  help='Input the questions each with the format of SimpleNamespace')

  # parser.add_argument('--codefile', type=str, default='Ouput file in the metrics folder')
  # parser.add_argument('--outfile',type=str,default='Output file in the metrics folder, version 0')
  #  parser.add_argument('--model',type=str,default='gpt-4o')

  args = parser.parse_args()
  main(args)