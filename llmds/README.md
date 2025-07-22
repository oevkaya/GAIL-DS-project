# Overview

This folder contains the core code for the GAIL project. The design was intended to 
be relatively modular so that new data science questions can be trained or 
implemented by the LLM agents. We provide a brief overview of each of the folders/files:

- params: Contains the functions
    - 'load_params': Reads questions from jsonl files and keep in SimpeNamespace.
    - 'input_dataset': Split into different datasets, specific to ids, questions, concepts and difficulty levels.
    - 'data_name_mapping': Maps the usually used dataset name to the corresponding filename

- 'scripts': Contains the scripts to run all simulation experiments. 
    - 'automation': Covers two functions of multi-round experiments, one for running each of a selected 
    set of questions specific to one dataset, the other for running selected sequential questions specific to 
    one dataset. 

    - 'collect_results': Extracts text and Python code parts from the experiment outcomes, and separatedly store.
    Combines the following details into one jsonl file for a dataset: each question id, round, thread id, status 
    (completed or failed), runtime, number of words for text, number of tokens for text, reasoning part, and outcome. 
    For sequential questions, one additional step is to separate the outcomes per question. The features include

    {'id', 'round', 'thread_id','words','tokens','status','runtime','reasoning'}

    - 'evaluation': Combines the results from above with the corresponding input metrics (number of words or tokens 
    for the question input). Computes the metrics of Jaccard similarity and verbosity ratio. Compute the accuracy of 
    outcomes compared with GTs. 
