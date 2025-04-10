# GAIL-project-Examples
Prompt curation from courses and open source examples

## Data science tasks types from different sources

1. [InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks](https://arxiv.org/pdf/2401.05507)

Short notes: In this paper, we introduce InfiAgent-DABench,
the first benchmark specifically designed to evaluate LLM-based agents on data analysis tasks.
These tasks require agents to end-to-end solving
complex tasks by interacting with an execution
environment. This benchmark contains DAEval,
a dataset consisting of 257 data analysis questions derived from 52 CSV files, and an agent
framework which incorporates LLMs to serve as
data analysis agents for both serving and evaluation. Since data analysis questions are often
open-ended and hard to evaluate without human
supervision, we adopt a format-prompting technique to convert each question into a closed-form
format so that they can be automatically evaluated.

In this paper, we build InfiAgent-DABench for data analysis
tasks, which includes 257 questions associated with 52 CSV
files, covering realistic data analysis demands and a wide
range of domains. We crawl CSV files from GitHub and
instruct GPT-4 to generate open-ended questions based on
the file and several key concepts for data analysis obtained
from expert interviews.

## Tasks types

From Table 12: 
Concepts Summary Statistics 41 17 32 90
Feature Engineering 3 14 33 50
Correlation Analysis 10 32 30 72
Machine Learning 0 0 19 19
Distribution Analysis 21 23 20 64
Outlier Detection 5 20 10 35
Comprehensive Data Preprocessing

----------------------------------------------------

## Our Experimental setting 

## Our Task types and related questions at different levels 

## Our metrics to evaluate the specific task outcome 


