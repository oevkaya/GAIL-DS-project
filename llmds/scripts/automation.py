import os
import json
import time
from openai import OpenAI

def multi_round_assistant(client,Qs,dinput,model,file_id,Ns,path,ks):
  """Run multi-round experiments on each question of the dataset
  """
#   Qs = dinput['ids']; 
  Q_num = len(Qs)
  content_set = dinput['questions']

  #---------------create an assistant---------------
  assistant = client.beta.assistants.create (
    name = "Question and Code Assistant",
    instructions ="""
    You are a specialized assistant for iterative data‐science tasks. Every time the user asks a question or provides data.
    Answer each question.
    You will return a JSON object with one key: `"outcome"`, which is a string or JSON array describing the results.
    """,
    model = model,
    tools = [{'type': 'code_interpreter'}],
    tool_resources={ 
      'code_interpreter': {
        'file_ids': [file_id]
        }
    })

  assistant_id = assistant.id

  #---------------run multi-round simulations---------------
  for Q in range(Q_num):
    # create an empty text file
    text_file = f"{path}/Q{Qs[Q]}.txt"
    if not os.path.exists(text_file):
      with open(text_file, "w") as f:
        pass
    
    # create an empty image folder if necessary
    # if concept_set[Q] == 'Data Visualization':
    image_folder = f"{path}/Q{Qs[Q]}_image"
    if not os.path.exists(image_folder):
      os.makedirs(image_folder, exist_ok=True)

    for k in range(Ns[Q]):
      if k % 10 == 0:
        print(f"Round: {k}")

      start_time = time.time()
      #------------------Create a thread and run------------------
      thread = client.beta.threads.create()
      message = client.beta.threads.messages.create(
        thread_id = thread.id,
        role = 'user',
        content=content_set[Q] + ".Provide a complete Python snippet ready to run."
        )
      thread_id = thread.id

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
            
            with open(f"{image_folder}/{k+ks[Q]}.png", "wb") as file:
               file.write(image_data_bytes)

          
      with open(text_file, "a", encoding="utf-8") as f:
        f.write("-"*100)
        f.write(f"\nRound {k+ks[Q]} with thread_id: {thread.id}\n")
        f.write(f"\nimage_id: {image_id}\n")
        f.write(f"\nStatus: {status}, Runtime: {runtime}\n")

        for message in reversed(list(messages)):
           for content in message.content:
             if content.type == 'text':
               f.write("-"*50)
               f.write(f"\nRole: {message.role}\n")
               f.write(f"\n{content.text.value}\n\n")
    
  
  return assistant_id

def sequential_question_assistant():
  pass