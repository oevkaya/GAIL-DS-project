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

2.[DA-Code: Agent Data Science Code Generation Benchmark for Large Language Models](https://arxiv.org/pdf/2410.07331)


We present an overview of DA-Code’s data statistics, showcasing its structure and variety of tasks. DA-Code contains 500 tasks in total, categorized into Data Wrangling (DW), Machine Learning (ML), and Exploratory Data Analysis (EDA).
The ML tasks are comprised of sub-tasks such as Classification, Regression, and Clustering. EDA includes Visualization, Statistical Analysis, Data Insights, and Data Manipulation, while DW encompasses tasks such as Data Loading, Cleaning, and Transformation.

To achieve this goal, we introduce DA-Code, a
benchmark for evaluating LLM data analysis ability, with carefully defined task scenarios. DA-Code
contains 500 complex task examples, originating
from real, challenging data analysis tasks, encompassing three main categories: data wrangling
(DW), machine learning (ML) and exploratory
data analysis (EDA). It covers the entire data analysis pipeline. Data wrangling includes a variety
of tasks such as data loading, data cleaning, and
data merging, specifically targeting raw data in
files and databases. EDA aims to gain insights and
analysis using the given information and resources.
It includes a wide variety of data analysis tasks
using programming languages such as SQL and
Python to get insights from data.

----------------------------------------------------


## Our Experimental setting 

Main goal is to examine the performance of DA plugin with certain GPT 4 version such as 4o or 4o mini version regarding its price 


## Our Task types and related questions at different levels 

- Data Undertanding and Wrangling
- Data Cleaning and Pre-processing
- Data Summary Statistics and Interpretations
- Data Transformations

- Data Visualization - EDA 

- Statistical Models under supervised , unsupervised learning type of models
  - Supervised Learning 
    - Linear Regression modeling
    -  Variations of regression (Lasso or Ridge)
    - Classification models such as Logistic Regression model
    - Tree based model
    - Model comparison type questions
  - Unsupervised Learning
    - Clustering methods
    - Dimensionality Reductions (PCA)

- Statistical Hypothesis testing related questions 

## Our metrics to evaluate the specific task outcome 


