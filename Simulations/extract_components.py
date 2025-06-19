""" 
This file extracts components from the main text file 
and constructs the components into one json file,

Target format:
Each question has one id and the collection of 
multi-round outcomes, like

{"id":1, "results":{{"round":1, "outcomes":"", ...}...} }

or 

{"id":1, "round":1}
{"id":1, "round":2}
{"id":1, "round":3}

"""

# Required input
Q = 0

code_style = "Python"
model_choice = "gpt-4o"

text_file = f"Simulations/output/UK-visitor-numbers/Q{Q}.txt"
response = f"Simulations/responses/Q{Q}.jsonl"

# codefile = f"Simulations/Q{Q}/Q{Q}.py"
# reasonfile = f"Simulations/Q{Q}/Q{Q}_reasoning.txt"
# outcomefile = f"Simulations/Q{Q}/Q{Q}_outcome.txt"
# image_folder = f"Q{Q}_image"


#---------------Functions---------------
import re
import tiktoken
import json
import os

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

def count_tokens(text, model=model_choice):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# def extract_brief_answer(text: str):
#     """
#     Extracts a likely numeric result from a sentence.
#     Looks for numbers followed by units or keywords.
#     """
#     pattern = re.compile(
#         r"(?i)\b(?:the\s+dataset\s+.*?|\bthere\s+are\b|\btotal\s+of\b|found\b|is\b|includes\b)?\s*(\d{1,5}(?:[.,]\d+)?(?:\s+\w+){0,4})[.]",
#         re.IGNORECASE
#     )

#     matches = pattern.findall(text)
#     return matches[0].strip() if matches else None


def extract_outcome(text):
    """
    Extracts phrases like 'contains 348', 'are 25', or 'has 109' from assistant responses.
    Returns the first match or None.
    """
    pattern = r"(?:contains|are|has|include|comprise|involve)\s+(about\s+)?(\d+[,\d]*)(\s+\w+)?"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).strip() if match else None



def extract_assistant_response(Q, section,version, jsonfile=None,codefile=None,reasonfile=None,outcomefile=None):

    code = extract_code_blocks(section)
    code_part = code[0] if code else None

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
          content_token += count_tokens(block)
          content_word += len(block.split())

          reasoning.append(extract_reasoning(block))

    #--------------Save each component--------------
    if version == "json":
      
      json_block = {
         "id":Q, 
         "round":elements['round'],
         "thread_id":elements['thread_id'],
         "status":elements['status'],
         "runtime":elements['runtime'],
         "words":content_word,
         "tokens":content_token,
         "code":code_part,
         "reasoning": reasoning,
         "outcome": outcome
    }   
    
      with open(jsonfile, "a", encoding="utf-8") as f:
       json.dump(json_block, f, ensure_ascii=False)
       f.write("\n")
      
    if version == "origin":
       
       # Save code block
       if code_part:
          with open(codefile, "a", encoding="utf-8") as f:
            f.write("#"*50)
            f.write(f"\n#Round {elements['round']} with threat_id: {thread_id}\n")
            f.write(code_part)

       # Save reasoning block
       # if reasoning:
       with open(reasonfile, "a", encoding="utf-8") as f:
        f.write("\n"+"-"*50)
        f.write(f"\nRound {elements['round']} with thread_id: {elements['thread_id']}")
        f.write(f"\nStatus: {elements['status']}, Runtime: {elements['runtime']}")
        f.write(f"\nTokens: {content_token}, Word: {content_word}\n")
        f.write("\n".join(reasoning))

       # Save outcome block
       if outcome:
        with open(outcomefile, "a", encoding="utf-8") as f:
          f.write("-"*50)
          f.write(f"\nRound {elements['round']} with outcome: {outcome} \n")

    #--------------Metrics collection--------------
    # Directly output the metrics for evaluation

    # return elements


# from collections import defaultdict
# dict_elements = defaultdict(list)

# dict_elements = []


# with open(text_file, "r", encoding="utf-8") as f:
#   content = f.read()

#   # Find the section for each thread_id, separated by 100-dash separator
#   sections = [section.strip() for section in content.split("-" * 100) if section.strip()]

#   for section in sections:
#     id_match = re.search(r"Round (\d+)\s+with thread_id:\s+(\w+)", section)
#     if id_match:
#         round_num = int(id_match.group(1))
#         thread_id = id_match.group(2)

#     # Choose the version of json/jsonl, or separate code/reasoning/outcomes files. 
#     extract_assistant_response(Q, section,"json", jsonfile=response)
#     # extract_assistant_response(Q, section,"origin", codefile, reasonfile,outcomefile)

#     # for e, y in elements.items():
#     #    dict_elements[e].append(y)

#     # dict_elements.append(elements)

#----------------------------Version 2----------------------------

def extract_code_for_evaluation(Q,section,round_num,thread_id,codefile=None):
  code = extract_code_blocks(section)
  code_part = code[0] if code else None
    
  # Save code block
  if code_part:
    with open(codefile, "a", encoding="utf-8") as f:
      f.write("#"*50)
      f.write(f"\n#Question {Q}, Round {round_num} with threat_id: {thread_id}\n")
      f.write(code_part)


def extract_response_for_evaluation(Q, section,jsonfile=None):

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
          content_token += count_tokens(block)
          content_word += len(block.split())

          reasoning.append(extract_reasoning(block))
    
    #--------------Save each component--------------
    if not os.path.exists(jsonfile):
      with open(jsonfile, "w") as f:
        pass

    if not os.path.exists(codefile):
      with open(codefile, "w") as f:
        pass
      
    json_block = {
       "id":Q, 
       "round":elements['round'],
       "thread_id":elements['thread_id'],
       "status":elements['status'],
       "runtime":elements['runtime'],
       "words":content_word,
       "tokens":content_token,
       "reasoning": reasoning,
       "outcome": outcome
    }   
    
    with open(jsonfile, "a", encoding="utf-8") as f:
      json.dump(json_block, f, ensure_ascii=False)
      f.write("\n")
      
def format_number(x):
    return int(x) if x == int(x) else str(x)

import os
import glob
input_folder = "Simulations/output/UK-visitor-numbers"
input_files = glob.glob(os.path.join(input_folder, '*.txt'))

output_file = "Simulations/metrics/UK-visitor-numbers.jsonl"
codefile = f"Simulations/metrics/visitor.py"

for text_file in input_files:
    print(text_file)

    match = re.match(r"Q(\d+(?:\.\d+)?)\.txt", os.path.basename(text_file))
    Q = format_number(float(match.group(1)))
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
          extract_response_for_evaluation(Q, section,jsonfile=output_file)

          extract_code_for_evaluation(Q,section,round_num,thread_id,codefile=codefile)