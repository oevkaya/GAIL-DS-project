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

### Comprehensive Data Preprocessing

| Topic                          | Col 1 | Col 2 | Col 3 | Total |
| ------------------------------ |-----:|-----:|-----:|-----:|
| Concepts Summary Statistics    |   41 |   17 |   32 |   90 |
| Feature Engineering            |    3 |   14 |   33 |   50 |
| Correlation Analysis           |   10 |   32 |   30 |   72 |
| Machine Learning               |    0 |    0 |   19 |   19 |
| Distribution Analysis          |   21 |   23 |   20 |   64 |
| Outlier Detection              |    5 |   20 |   10 |   35 |


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

we have more papers on that to write down the Related Research section of our report / paper

----------------------------------------------------


## Our Experimental setting 

Main goal is to examine the performance of DA plugin with certain GPT 4 version such as 4o or 4o mini version regarding its price 

So far, we are doing experiments on the model = "gpt-4o" but there could be some alternative models on specific tasks to make a comparison such as gpt-4o-mini, o4-mini or evengpt-3.5-turbo
This will depend on the pricing, the selected list of tasks to compare, not the whole set of questions


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

### Source of Questions/Prompts

- HDSR paper related implementations; laptop and house price data sets and model related questions that we tried already + certain data viz exercises based on the reviewer's suggestions
- IDS course weekly lab exercises, Homework assigments varying on different data sets in .csv file format and data sets from the certain R package directly
- IDS quiz examples including interpretational type of questions without any specific data file (Not in a higher priority)
- Some Statistical related questions from our Statistics Year 2 from weekly labs / quiz exercises from previous years (subject to ST's confirmation)
- Other open source data sets used in other papers and also shared via Github already (open to public), adaptations of some of them can be useful for the enrichment of the our data set  

## Our metrics to evaluate the specific task outcome 

This part is subject to change based on the task we are exploring. The goal is to extract certain components of the response to parse further and do additional analysis at that moment

One simple example can be considered in this format;

{
  "reasoning": "I will perform linear regression on GDP per capita vs. life expectancy to quantify their relationship using R-squared.",
  "code": "```python\nimport pandas as pd\nfrom sklearn.linear_model import LinearRegression\n\ndf = pd.read_csv('Happiness_rank.csv')\nX = df[['Economy (GDP per Capita)']].values.reshape(-1,1)\ny = df['Health (Life Expectancy)'].values\nmodel = LinearRegression().fit(X, y)\nr2 = model.score(X, y)\nprint(r2)\n```",
  "outcome": {
    "summary": "The regression yields an R-squared of approximately 0.67, indicating a poor fit.",
    "@coefficient_determination": 0.67,
    "@model_fit": "poor fit"
  }
}






