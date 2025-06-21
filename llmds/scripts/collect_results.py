import pickle
import argparse
import re
import tiktoken
import json
import os
import glob
import numpy as np
from types import SimpleNamespace

#------------------Extraction------------------
def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def extract_code_blocks(text):
    pattern = r"```(?:python)?\n(.*?)```"
    # pattern = r"```(\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)

def extract_reasoning(text):
  """ 
  Remove Python/R/other language code and the header from text 
  """
  pattern = r"```(?:python)?\n(.*?)```"
  # pattern = r"```(\w+)?\n(.*?)```"
  header = "Role: assistant"

  def clean_block(text):
     text = re.sub(pattern, '', text, flags=re.DOTALL)
     text = re.sub(header, '', text).strip()

     return text

  if isinstance(text, list):
    return [clean_block(t) for t in text]
  else:
    return clean_block(text)
  
def extract_metadata(header_block):
    """Extract round ID, thread ID, runtime, and status from the first block."""
    round_match = re.search(r"Round\s+(\d+)", header_block)
    thread_match = re.search(r"thread_id:\s*(\S+)", header_block)
    runtime_match = re.search(r"Runtime:\s*([\d.]+)", header_block)
    status_match = re.search(r"Status:\s*(\w+)", header_block)

    return {
        "round": int(round_match.group(1)) if round_match else None,
        "thread_id": thread_match.group(1) if thread_match else None,
        "runtime": float(runtime_match.group(1)) if runtime_match else None,
        "status": status_match.group(1) if status_match else None
    }


def extract_outcome(text):
    """
    Extracts phrases like 'contains 348', 'are 25', or 'has 109' from assistant responses.
    Returns the first match or None.
    """
    pattern = r"(?:contains|are|has|include|comprise|involve)\s+(about\s+)?(\d+[,\d]*)(\s+\w+)?"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).strip() if match else None


def extract_code_for_evaluation(Q,section,round_num,thread_id,codefile=None):
  code = extract_code_blocks(section)
  code_part = code[0] if code else None
  
  if not os.path.exists(codefile):
    with open(codefile, "w") as f:
       pass

  # Save code block
  if code_part:
    with open(codefile, "a", encoding="utf-8") as f:
      f.write("#"*50)
      f.write(f"\n#Question {Q}, Round {round_num} with threat_id: {thread_id}\n")
      f.write(code_part)


def extract_response_for_evaluation(Q, section,model,jsonfile=None):

    # Split responses by 50-dash separator
    blocks = [block.strip() for block in section.split("-" * 50) if block.strip()]

    elements = extract_metadata(blocks[0])
    outcome = extract_outcome(blocks[-1])
    reasoning = []

    content_token = 0
    content_word = 0

    for block in blocks[1:]:
      # Skip user blocks
      if block.startswith("Role: user"):
        continue

      if block.startswith("Role: assistant"):
        #   block_clean = clean_block(block)
          content_token += count_tokens(block,model)
          content_word += len(block.split())

          reasoning.append(extract_reasoning(block))
    
    #--------------Save each component--------------
    if not os.path.exists(jsonfile):
      with open(jsonfile, "w") as f:
        pass
    
    out_block = SimpleNamespace(
       id=Q, 
       round=elements['round'],
       thread_id=elements['thread_id'],
       status=elements['status'],
       runtime=elements['runtime'],
       words=content_word,
       tokens=content_token,
       reasoning=reasoning,
       outcome=outcome
    )
    with open(jsonfile, "a", encoding="utf-8") as f:
       f.write(json.dumps(out_block.__dict__)+"\n")
       
    # json_block = {
    #    "id":Q, 
    #    "round":elements['round'],
    #    "thread_id":elements['thread_id'],
    #    "status":elements['status'],
    #    "runtime":elements['runtime'],
    #    "words":content_word,
    #    "tokens":content_token,
    #    "reasoning": reasoning,
    #    "outcome": outcome
    # }   
    
    # with open(jsonfile, "a", encoding="utf-8") as f:
    #   json.dump(json_block, f, ensure_ascii=False)
    #   f.write("\n")
      
def find_number(text_file):
    match = re.match(r"Q(\d+(?:\.\d+)?)\.txt", os.path.basename(text_file))
    x = float(match.group(1))
    return int(x) if x == int(x) else str(x)

# def extract_components(input_folder,path,dataname):
def main(args):
   print(f"Extract components for {args.dataname}")

   infiles = glob.glob(os.path.join(args.input_folder, '*.txt'))
   
   outfile = f"{args.path}/{args.dataname}0.jsonl"
   codefile = f"{args.path}/{args.dataname}.py"

   for text_file in infiles:
      print(text_file)

    #   match = re.match(r"Q(\d+(?:\.\d+)?)\.txt", os.path.basename(text_file))
      Q = find_number(text_file)

      with open(text_file, 'r') as f:
        content = f.read()

        # Find the section for each thread_id, separated by 100-dash separator
        sections = [section.strip() for section in content.split("-" * 100) if section.strip()]

        for section in sections:
          id_match = re.search(r"Round (\d+)\s+with thread_id:\s+(\w+)", section)
          if id_match:
            round_num = int(id_match.group(1))
            thread_id = id_match.group(2)

          # Choose the version of json/jsonl, or separate code/reasoning/outcomes files. 
          extract_response_for_evaluation(Q, section,args.model,jsonfile=outfile)

          extract_code_for_evaluation(Q,section,round_num,thread_id,codefile=codefile)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_folder', type=str, default='Simulations/output/weatherAUS')
   parser.add_argument('--path', type=str, default='Simulations/metrics/weatherAUS')
   parser.add_argument('--dataname',type=str,default='weatherAUS')
   parser.add_argument('--model',type=str,default='gpt-4o')

   args = parser.parse_args()
   main(args)