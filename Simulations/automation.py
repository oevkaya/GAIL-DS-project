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

data = 'laptop_data_cleaned.csv'
file_id = 'file-2JxoJLhwmf8KZfypqywhw3'
folder_path = 'Simulations/output/laptop'


with open(f"{folder_path}/input.json", "r", encoding="utf-8") as f:
  input = json.load(f)

content_set = input['content_set']
concept_set = input['concept_set']

Q_num = len(concept_set)

# content_set = [
#     'Give me the statistical summary for the given data set',
#     'Create the suitable data visualization to summarize the price distribution and its relationship with other features',
#     'Implement the linear regression for the price variable for the given data set',
#     'Test and Interpret the performance of the fitted regression model based on the model diagnostic plots',
#     """Implement different regression models as the alternative of fitted linear regression model for the price variable for the given data set. 
# Compare the performance of the fitted models""",
#     """Fit a regression tree model for the price variable for the given data set and test its performance. 
# Tune the model hyperparameters by using the cross-validation approach""",
#     """Consider the classification problem and use models like logistic regression, decision trees, or neural networks 
# for laptops into different price ranges (e.g., budget, mid-range, premium)"""
#     ]

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

print(assistant)

# run multi-round simulations
for Q in range(Q_num):
    # create an empty text file
    text_file = f"{folder_path}/Q{Q}.txt"
    with open(text_file, "w") as f:
      pass

    # create an empty image folder if necessary
    if concept_set[Q] == 'Data Visualization':
        image_folder = f"{folder_path}/Q{Q}_image"
        os.makedirs(image_folder, exist_ok=True)
        print(f"Folder '{image_folder}' created (or already exists).")


    for k in range(N):
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
        thread_id = thread.id,
        assistant_id = assistant.id,
      )

      run = client.beta.threads.runs.retrieve(
        thread_id = thread.id,
        run_id = run.id
      )

      while run.status not in ["completed", "failed"]:
        run = client.beta.threads.runs.retrieve(
        thread_id = thread.id,
        run_id = run.id
        )
        time.sleep(10)

      #------------------Result------------------
      runtime = time.time() - start_time
      status = run.status
      messages = client.beta.threads.messages.list(
        thread_id = thread.id,
      )
    
      #------------------Result Extraction------------------
      if concept_set[Q] == 'Data Visualization':
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

      else:
        with open(text_file, "a", encoding="utf-8") as f:
          f.write("-"*100)
          f.write(f"\nRound {k} with thread_id: {thread_id}\n")
          f.write(f"\nStatus: {status}, Runtime: {runtime}\n")

          for message in reversed(list(messages)):
            content = message.content[0].text.value
            f.write("-"*50)
            f.write(f"\nRole: {message.role}\n")
            f.write(f"\n{content}\n\n")
            
    