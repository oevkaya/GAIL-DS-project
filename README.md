# GAIL-project-Examples

This file summarizes the main aspects of the project. For further details and experimental results, please refer to the ??

----------------------------------------------------

## Our Experimental setting 

Main goal is to examine the performance of DA plugin with certain GPT 4 version such as 4o or 4o mini version regarding its price 

So far, we are doing experiments on the model = "gpt-4o" but there could be some alternative models on specific tasks to make a comparison such as gpt-4o-mini, o4-mini or evengpt-3.5-turbo
This will depend on the pricing, the selected list of tasks to compare, not the whole set of questions

### Change on the settings

- Fixing the prompt and the other details, use (i) model = "gpt-4o", (ii) model = "gpt-4o-mini", (iii) model = "4o-mini", also one option for vanilla LLM like without code interpreter can be tried (cheapest option for testing)
- Fixing the prompt and the model, use different temperature values by tweaking the "temperature = default" value, it can be like (0.2, 0.5, 1.0, 1.5) based on its original range
- Fixing the prompt and all other settings, use the prompt with programming language constraint or not such as "constraint: "Using the built-in functions and suitable R packages"' in the prompt or system instruction

Impact of zero shot vs one shot learning can be explored for some tasks only 

## Our Task types and related questions at different levels 

- Data Undertanding and Wrangling
- Data Cleaning, Pre-processing and Transformations
- Data Summary Statistics and Interpretations
  
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
- Bayesian regression

## Task labeling results (TBA)

| Topic                          | Easy | Medium | Hard | Total |
| ------------------------------ |-----:|-----:|-----:|-----:|
| Data Undertanding/Wrangling    |   ?? |   ?? |   ?? |   ?? |
| Data Summary / Intepretations  |   ?? |   ?? |   ?? |   ?? |
| Data Visualization             |   ?? |   ?? |   ?? |   ?? |

### Source of Questions/Prompts

- HDSR paper related implementations; laptop and house price data sets and model related questions that we tried already + certain data viz exercises based on the reviewer's suggestions
- IDS course weekly lab exercises, Homework assigments varying on different data sets in .csv file format and data sets from the certain R package directly
- IDS quiz examples including interpretational type of questions without any specific data file (Not in a higher priority)
- Some Statistical related questions from our Statistics Year 2 from weekly labs / quiz exercises from previous years (subject to ST's confirmation)
- Some questions from Bayesian Data Analysis and Multivariate Data Analysis courses to cover PCA, Clustering, Bayesian modelling as different style of tasks
- **NOT USED SO FAR: Other open source data sets used in other papers and also shared via Github already (open to public), adaptations of some of them can be useful for the enrichment of the our data set later**

### Possible format of the prompts

Question List: GAIL-DA-tasks-questions.jsonl file but will be expanded

- **id**: Unique identifier for each question.
- **question**: The description of the data analysis question.
- **concepts**: The concepts involved in the question.
- **format**: The format requirements for the output.
- **file_name**: The file name of the corresponding csv file.
- **level**: The difficulty level for each question.

Additional details like constraints: The constraints on the type of programming (using R instead of Python) or similar ones can be considered!
We will gather feedback from the audience for the **concepts** and the **level** information by asking them randomly allocated set of questions.

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

### Possible metrics to think about

The main metrics for the created responses can be listed under three subsections mainly 

#### General Properties

- **Verbosity**: General length for the number of tokens or words to count the verbosity of the generated response in general except the coding component. This will focus on only the text related part! So far we have number of words and tokens
- **Runtime** for the response creation for each iteration
- **Verbosity ratio (input / output)** in terms of either words or tokens: Can be considered! It is the ratio of number of input tokens (words) / number of output tokens (words) 

#### Course-Grained Metrics

- **Completion Ratio (CR)**: This can be two different values in general (i) 0: if the executed thread is failed/not completed, (ii) 1: if the executed thread completed with an acceptable outcome (either matching with ground truth or not). This can be measured for each prompt out of 100 trials based on the stored list of responses as a ratio
- **Response Accuracy (RA)** or **Accuracy of Response (AoR)**: Whether the reported result is matching with the ground truth or not (can be numeric or string, or vector etc.), it can take either 0 or 1 again. If the output is decimal value, or if the match appears partially it can be controlled during the comparison for numerical values!
  
- **Code Executability**: Whether the generated code is directly executable in a different environment or not, either taking 2 (it is executable and solving the anticipated problem), taking 1 (it is directly executable BUT NOT solving the problem) or 0 (not executable at all). 
  
- **Text Similarity**: Similarity measures such as Jaccard Index and other can be considered for the text part of the generated response to compare with each other or ground truth explanations for some tasks. Source: https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python. THIS CAN BE ADDED FOR SOME INTERPRETATION BASED QUESTIONS

#### Task-Specific Metrics

For each of the concepts we are considering, varying set of criterions can appear for the output evaluation in general. These task related ones can be listed under;

- Data Cleaning/Pre-processing
- Data Summary stats and interpretations (each of them, for each run, is either 0 and 1 so in total out of 5)
  - **Ratio of Accuracy** (will be calculated automatically via comparing GT value)
  - **Data Context recognition** Clearly define each variable, its units of measurement, and data source to ground the summary in context
  - **Appropriateness of Stats Chosen:** Were the most relevant statistics reported for the data type within the context of question (e.g., median for skewed data, mode for categorical)? 
  - **Accuracy of Interpretation:** Do the interpretations logically follow from the statistics? (e.g., not saying data is "normally distributed" based on wrong evidence) 
  - **Clarity of Interpretation:** Are the interpretations understandable, concise, and easy to follow or is there any confusing word selection or 

- Data Visualization Quality (each of them, for each run, is either 0 and 1 so in total out of 5)
  - **Data Viz Completeness**: If the requested data visualization is generated or not, it can take 1 (if they exists) or 0 (if not) otherwise. If this takes 0, nothing to compute indeed so this item is just the starting point so not included in the overall
  - **Aesthetic Mapping (Mapping Layer)**: Each data variable is unambiguously mapped to an appropriate aesthetic (e.g., a continuous variable to position, a categorical variable to color), and the mappings are clearly documented in a legend or caption.
  - **Geometric Object (Geom Layer)**: The chosen geom (e.g., geom_bar() for counts, geom_point() for scatter) matches the data structure and analytical intent given by the question, with no misuse of graphical primitives or not suitable geom selection
  - **Scales & Coordinate System (Scale/Coord Layers)**: Axes use appropriate, non‑truncated scales with meaningful breaks; any coordinate transform (e.g., log scale, coord_flip()) is applied suitably, all components are readable fully.
  - **Visualization-Interpretation matching**: The correct use of wordings and abbreviations in the created visual and the related interpretations. The visualization considers the use of created output while creating the interpretations (MIGHT NOT BE RELEVANT FOR SOME QUESTIONS)

- Statistical Modeling: It may change slightly in terms of which modeling approach we are applying; regression, classification, PCA or clustering. For each of them, we have the **Response Accuracy (RA) or Accuracy of Response (AoR)** as the first item (mentioned above), then the following specific evaluations are applied additionally to get a task specific numerical evaluation 

  For Linear Regression (all either 0 or 1);
  - **Ratio of Accuracy** (will be calculated automatically via comparing GT value)
  - **Correct model coefficient interpretation:** Understanding the magnitude and direction of the relationship between independent and dependent variables
  - **p-value recognition** Recognizing statistical significance on the model coefficients by reporting of p-values for each coefficient against a pre-specified α-level (commonly 0.05).
  - **Evaluation Metric Reported and Interpreted:** Assessing the model's goodness of fit and the proportion of variance explained by the model via R_squared, supplemented by predictive error metrics such as RMSE
  - **Interpretation of Results:** Interpreting the results within the context of the specific research question and the problem domain
 
  For Logistic Regression;
  - **Ratio of Accuracy** (will be calculated automatically via comparing GT value)
  - **Correct model coefficient interpretation:** Understanding the relationship between independent and dependent variables (via transforming log-odds coefficients into odds ratios)
  - **p-value recognition** Recognizing statistical significance on the model coefficients by reporting of p-values for each coefficient against a pre-specified α-level (commonly 0.05).
  - **Confusion Matrix recognition:** The confusion matrix details and derive class-specific error rates
  - **Interpretation of Results:** Interpreting the results within the context of the specific research question and domain

  For PCA (either 0 or 1 for each based on response);
  - **Number of Components Justified:** Is the decision on how many PCs to keep justified correctly?
  - **Standardization of Input Data:** Was scaling applied where needed before PCA?
  - **Correct Variance Explanation:** Are eigenvalues / explained variance ratios reported and understood?
  - **Proper Component Interpretation:** Are PCs interpreted clearly in terms of the concept of original variables?
  - **Visual evidences** Is the decision on the number of PCs supported by certain visuals like loading, biplot or scree plot
    
  For Clustering;
  - **Number of clusters justified:** Is the number of obtained clusters are justified correctly ?
  - **The methods consistency:** Is the requested method applied properly with the specification of certain parameters
  - **Cluster characteristics:** Are the obtained cluster characteristics such as cluster size, cluster profile mentioned or not
  - **Validation metrics:** Are the certain report metrics like silhouette scores, Davies-Bouldin index, or other relevant statistics are recognised
  - **Visual evidences:** Is the decision on the number of clusters supported by certain visuals like scatter plots, dendrograms
  
- Hypotesis testing (each of them either marked as 0 or 1, in total 5)
  - **Correct Interpretation of p-value and/or Confidence Interval:** Having the correct final decision based on the testing procedure
  - **Correct Test/Test name Chosen:** Recognition of the correct statistical testing / test name
  - **Null and Alternative Hypotheses Clearly Stated:** clearly and correctly stated hypothesis test components with correct notation
  - **Test Assumptions Recognition:** Correctly recognised the selected test assumptions 
  - **Contextual Conclusion:** Interpreting the final decision within the context of the specific research question

For each of above, we can create a list of different evaluation criterions to give a mark out of 5 so that each result may have numerical evaluation number! 

## How to conduct the experiments for new dataset or questions

To run multi-round experiments on a selected set of questions specific to one dataset, we provide a single file 
(*runner.py*) that can accommodate both independent questions and sequential questions. The adjustable setting is 
- model choice (default: 'gpt-4o')
- temperature (1.0)
- instruction for the AI assistant
- data-file-related features: dataname, filename, file_id, outfolder
- question-related features: selected questions (Qs), the number of rounds (Ns), and the number of already-done rounds (ks).

If all the questions are independent, try the function **multi_round_assistant**.
If the questions are sequentially related, try the alternative **sequential_question_assistant**.

After obtaining the experimental outcomes, we provide a single file (*evaluation.py*) that can help to collect results
and make evaluation. The adjustable setting is the same as above, except the instruction being no longer used. 