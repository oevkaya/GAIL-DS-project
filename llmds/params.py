import json
import csv
import re
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

def remove_space(text):
    txt_clean = []
    for txt in text:
        txt_clean.append(re.sub(r'\s*-\s*', '-', txt))
    return txt_clean

def smart_title(s):
    """Format the concepts, capitalizing the initials """
    exceptions = {"EDA", "CI", "PCA", "Lasso"}

    semantic_normalization = {
        "Logistic Regression Modeling": "Logistic Regression Model",
        "Regression Modeling Interpretations": "Regression Modeling Interpretation",
    }

    s = s.strip()
    # Split by spaces
    words = s.split()
    result = []

    for word in words:
        parts = word.split('-')  # Handle hyphenated words
        titled_parts = []
        for part in parts:
            upper_part = part.upper()
            if upper_part in exceptions:
                titled_parts.append(upper_part)
            else:
                titled_parts.append(part.capitalize())
        result.append('-'.join(titled_parts))

    titled = ' '.join(result)

    # Apply semantic normalization
    return semantic_normalization.get(titled, titled)

def csv_to_text(csv_path, txt_path):
    """(Used for file search) As csv files are not supported by vector store,
    we need to convert to txt for data loading. """
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Format rows into readable text
    lines = []
    header = rows[0]
    for row in rows[1:]:
        line = ", ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(header, row))
        lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as txtfile:
        txtfile.write("\n".join(lines))

data_name_mapping = {
    'aeroplane':'aeroplane.txt',
    'UK-visitor-numbers':'UK-visitor-numbers.csv',
    'evals':'evals.csv',
    'weatherAUS':'weatherAUS.csv',
    'instructional-staff':'instructional-staff.csv',
    'edibnb':'edibnb.csv',
    'gss16':'gss16_advfront.csv',
    'laptop_data_cleaned':'laptop_data_cleaned.csv',
    'duke-forest':'duke_forest.csv',
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