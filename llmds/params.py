import json
from types import SimpleNamespace

def load_params(jsonl_path: str):
    data = []
    buffer = ""
    invalid_blocks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip comment lines
            if line.startswith('##') or line.startswith('%'):
                continue

            # If line is empty and we have a JSON block collected
            if not line and buffer:
                try:
                    obj = SimpleNamespace(**json.loads(buffer))
                    data.append(obj)
                except json.JSONDecodeError as e:
                    invalid_blocks.append(buffer)
                
                buffer = "" # clean for the next block

            else:
                buffer += line # accumulate the json content

    # if invalid_blocks:
    #     print("Warn: Some blocks failed to parse")
    #     for block in invalid_blocks:
    #         print("\n...",block)
    
    return data


def input_dataset(data, filename):
    """Filter the input file for specific dataset,
    including id, question, concept, and level per question.
    """
    filenames = set([val.file_name for val in data])
    if filename in filenames:
        # print(f"Data file eixsts: {filename}")
        data_file = [item for item in data if filename in item.file_name]

        input = {
            "concepts": [val.concepts for val in data_file],
            "questions": [val.question for val in data_file],
            "ids": [val.id for val in data_file],
            "levels": [val.level for val in data_file]
        }

        return input
    else:
        raise ValueError("Invalid dataset!")

def call_question_by_ids(data,ids):
    return [val.question for val in data if val.id in ids]

def format_namespace(path):
   with open(path, 'r') as f:
      return [SimpleNamespace(**json.loads(line)) for line in f]

data_name_mapping = {
    'aeroplane':'aeroplane.txt',
    'UK-visitor-numbers':'UK-visitor-numbers.csv',
    'evals':'evals.csv',
    'weatherAUS':'weatherAUS.csv',
    'instructional-staff':'instructional-staff.csv',
    'edibnb':'edibnb.csv',
    'gss16':'gss16_advfront.csv',
    'laptop_data_cleaned':'laptop_data_cleaned.csv',
    'duke_forest':'duke_forest.xlsx',
    'Stats_diamonds':'Stats_diamonds.xlsx',
    'ggplot::diamonds':'diamonds.csv',
    'council_assessments':'council_assessments.csv',
    'mouse':'mouse.txt',
    'cherryblossom':'cherryblossom_run17.csv'
}

fileid_name_mapping = {
    'aeroplane':'file-Dz6rCnDN1WxfBh8L2t6d63',
    'UK-visitor-numbers':'UK-visitor-numbers.csv',
    'evals':'file-6GJ3f1PURyGNrWWs4qyUnJ',
    'weatherAUS':'file-XAij9D6dfvcPVkX2jLgCjC',
    'instructional-staff':'file-5riwCDAHXme7U6mfNZwCLe',
    'edibnb':'edibnb.csv',
    'gss16':'gss16_advfront.csv',
    'laptop_data_cleaned':'file-NoWy6mgGZpqszQtXdyd2Ys',
    'duke_forest':'file-HPzoG6QmUrABfAzMejmkrh',
    'Stats_diamonds':'Stats_diamonds.xlsx',
    'ggplot::diamonds':'file-Tqk79Wvgu7KSkC1WKtpGAb',
    'council_assessments':'council_assessments.csv',
    'mouse':'mouse.txt',
    'cherryblossom':'file-CkaqWNY14y7jXapdEbtxYT'
}