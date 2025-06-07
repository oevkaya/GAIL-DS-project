""" 
Create an automatic workflow of running multi-round simulations for 
a set of questions associated with the specific dataset. 
"""
import os
import json
import time
from openai import OpenAI
from pprint import pprint

N = 50 # number of rounds

data = 'UK-visitor-numbers.csv'
file_id = 'file-LB38GMvxj7SfRMPrenaBR1'
folder_path = 'Simulations/output/UK-visitor-numbers'


with open(f"{folder_path}/input.json", "r", encoding="utf-8") as f:
  input = json.load(f)

content_set = input['content_set']
concept_set = input['concept_set']

Q_num = len(concept_set)
Q_remain = [i for i in range(Q_num) if i not in {0, 1, 5, 7}]

openai_key="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"

client=OpenAI(
    api_key=openai_key
)

# create an assistant
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

# run multi-round simulations
for Q in Q_remain:
    # create an empty text file
    text_file = f"{folder_path}/Q{Q}.txt"
    with open(text_file, "w") as f:
      pass
    

    # create an empty image folder if necessary
    # if concept_set[Q] == 'Data Visualization':
    image_folder = f"{folder_path}/Q{Q}_image"
    os.makedirs(image_folder, exist_ok=True)
    print(f"Folder '{image_folder}' created (or already exists).")


    for k in range(N):
    #   print(f"Round: {k}")
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
    #   if concept_set[Q] == 'Data Visualization':
    #       image_id = None
    #       for message in reversed(list(messages)):
    #         for content in message.content:
    #            if content.type == 'image_file':
    #              image_id = content.image_file.file_id
    #              image_data = client.files.content(image_id)
    #              image_data_bytes = image_data.read()

    #              with open(f"{image_folder}/{k}.png", "wb") as file:
    #                 file.write(image_data_bytes)

          
    #       with open(text_file, "a", encoding="utf-8") as f:
    #         f.write("-"*100)
    #         f.write(f"\nRound {k} with thread_id: {thread.id}\n")
    #         f.write(f"\nimage_id: {image_id}\n")
    #         f.write(f"\nStatus: {status}, Runtime: {runtime}\n")

    #         for message in reversed(list(messages)):
    #            for content in message.content:
    #              if content.type == 'text':
    #                f.write("-"*50)
    #                f.write(f"\nRole: {message.role}\n")
    #                f.write(f"\n{content.text.value}\n\n")

    #   else:
    #     with open(text_file, "a", encoding="utf-8") as f:
    #       f.write("-"*100)
    #       f.write(f"\nRound {k} with thread_id: {thread_id}\n")
    #       f.write(f"\nStatus: {status}, Runtime: {runtime}\n")

    #       for message in reversed(list(messages)):
    #         content = message.content[0].text.value
    #         f.write("-"*50)
    #         f.write(f"\nRole: {message.role}\n")
    #         f.write(f"\n{content}\n\n")
            


response = client.beta.assistants.delete(assistant.id)