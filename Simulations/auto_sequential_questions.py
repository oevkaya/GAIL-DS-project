""" 
Create an automatic workflow of running multi-round simulations for 
sequential decision-making with the specific dataset. 

Use a single thread for the sequential questions.
"""
import os
import json
import time
from openai import OpenAI
from pprint import pprint

Q = 14

N = 10 # number of rounds

########################################################

data = 'instructional-staff.csv'
file_id = 'file-XtRrHxSoqJsT7Jr1Xw1bUV'
folder_path = 'Simulations/output/instructional-staff'


with open(f"{folder_path}/input.json", "r", encoding="utf-8") as f:
  input = json.load(f)

content_set = input['content_set']
concept_set = input['concept_set']
##########################################``

Q_num = len(concept_set)

openai_key="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"

client=OpenAI(
    api_key=openai_key
)

assistant = client.beta.assistants.create (
    name = "Question and Code Assistant",
    instructions ="""
    You are a specialized assistant for iterative data‐science tasks. Every time the user asks a question or provides data.
    Answer each question. Also provide a runnable Python snippet.
    You will return a JSON object with one key: `"outcome"`, which is a string or JSON array describing the results.
     """,
    model = "gpt-4o",
    tools = [{'type': 'code_interpreter'}],
    tool_resources={
        'code_interpreter': {
            'file_ids': [file_id]

        }
    }
)

assistant_id = assistant.id
print(assistant)

# create an empty text file
text_file = f"{folder_path}/Q{Q}.txt"
with open(text_file, "w") as f:
  pass

# create an empty image folder if necessary
image_folder = f"{folder_path}/Q{Q}_image"
os.makedirs(image_folder, exist_ok=True)
print(f"Folder '{image_folder}' created (or already exists).")


# run multi-round simulations

for k in range(N):
  if k % 5 == 0:
    print(f"Round: {k}")

  start_time = time.time()
  #------------------Create a thread for each question------------------
  thread = client.beta.threads.create()
  thread_id = thread.id


  for Q in range(Q_num):  
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = 'user',
        content=content_set[Q] + ".Please answer each question in order. Provide a complete Python snippet ready to run."
        )
    run = client.beta.threads.runs.create(
      thread_id = thread_id,
      assistant_id = assistant_id,
    )

    run = client.beta.threads.runs.retrieve(
      thread_id = thread_id,
      run_id = run.id
    )

    while run.status not in ["completed", "failed"]:
      run = client.beta.threads.runs.retrieve(
      thread_id = thread_id,
      run_id = run.id
      )
      time.sleep(10)
    
    #------------------Result------------------
    runtime = time.time() - start_time
    status = run.status
    messages = client.beta.threads.messages.list(
      thread_id = thread_id,
    )


    #------------------Result Extraction------------------
    image_id = None
    for message in reversed(list(messages)):
      for content in message.content:
        if content.type == 'image_file':
          image_id = content.image_file.file_id
          image_data = client.files.content(image_id)
          image_data_bytes = image_data.read()
            
          with open(f"{image_folder}/{k}.png", "wb") as file:
            file.write(image_data_bytes)

          
    with open(text_file, "a", encoding="utf-8") as f:
      f.write("-"*100)
      f.write(f"\nRound {k} with thread_id: {thread.id}\n")
      f.write(f"\nimage_id: {image_id}\n")
      f.write(f"\nStatus: {status}, Runtime: {runtime}\n")

      for message in reversed(list(messages)):
        for content in message.content:
           if content.type == 'text':             
             f.write("-"*50)
             f.write(f"\nRole: {message.role}\n")
             f.write(f"\n{content.text.value}\n\n")


response = client.beta.assistants.delete(assistant.id)

###########################
#############################################