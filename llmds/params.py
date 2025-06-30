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
    'diamonds':'ggplot::diamonds',
    'council_assessments':'council_assessments.csv'
}