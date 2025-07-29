##################################################
#Question 26.0, Round 0 with threat_id: thread_9VO1mSYoaX1R0gbctCfYBf6b
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame named 'data' with the necessary columns
# Replace 'your_file_path.csv' with the path to your dataset
data = pd.read_csv('your_file_path.csv')

# Convert gender to a categorical variable
data['gender'] = pd.Categorical(data['gender'])

# Define the independent variables (beauty rating and gender)
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variables

# Define the dependent variable (average professor evaluation score)
y = data['score']

# Add a constant to the independent variables matrix (for the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the intercept and the slopes
intercept = score_bty_gender_fit.params['const']
slope_bty_avg = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Adjust according to your dummy variable naming

print(f"Intercept (constant term): {intercept}")
print(f"Slope for beauty rating: {slope_bty_avg}")
print(f"Slope for gender (Male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='bty_avg', y='score', hue='gender', 
                jitter=True, palette='Set1', alpha=0.6)

plt.title('Scatterplot of Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 0 with threat_id: thread_B7ljiWEAXNKzCvCcOPibbU1S
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Make sure to replace 'your_data.csv' with the path to your actual dataset
data = pd.read_csv('your_data.csv')

# Model for score vs beauty for both genders
X = data['beauty']
y = data['score']
X = sm.add_constant(X)  # Adding a constant
model = sm.OLS(y, X).fit()

# Print summary of model
print(model.summary())

# Extract R-squared to find the percent of variability explained by the model
r_squared = model.rsquared
print(f"Percent of variability explained by the model: {r_squared * 100:.2f}%")

# Create subsets for male and female professors
male_data = data[data['gender'] == 'male']
female_data = data[data['gender'] == 'female']

# Fit model for male professors
X_male = male_data['beauty']
y_male = male_data['score']
X_male = sm.add_constant(X_male)
model_male = sm.OLS(y_male, X_male).fit()

# Fit model for female professors
X_female = female_data['beauty']
y_female = female_data['score']
X_female = sm.add_constant(X_female)
model_female = sm.OLS(y_female, X_female).fit()

# Get equations of the lines
male_intercept, male_slope = model_male.params
female_intercept, female_slope = model_female.params
print(f"Male professors line: score = {male_intercept:.2f} + {male_slope:.2f} * beauty")
print(f"Female professors line: score = {female_intercept:.2f} + {female_slope:.2f} * beauty")

# Plotting the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='beauty', y='score', hue='gender', data=data)
plt.plot(male_data['beauty'], male_intercept + male_slope * male_data['beauty'], color='blue', label='Male Regression Line')
plt.plot(female_data['beauty'], female_intercept + female_slope * female_data['beauty'], color='red', label='Female Regression Line')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.title('Beauty vs Score by Gender')
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 1 with threat_id: thread_m5CJvXTBQNnXiZiTV4LXPsxZ
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame named 'df' with columns 'score', 'bty_avg', and 'gender'.
# Load your data into df
# df = pd.read_csv('your_data_file.csv')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variable
y = df['score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Male' is the dummy created

print(f"Intercept: {intercept}")
print(f"Slope (beauty rating): {slope_bty}")
print(f"Slope (gender - Male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', data=df, hue='gender', jitter=True, palette='Set1')
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 1 with threat_id: thread_HXloVsB4xVCscg66n8IfrnVQ
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame creation - Replace this with your actual dataset
# df = pd.read_csv('your_dataset.csv')
# Example data creation
data = {
    'beauty': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'score': [2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
    'gender': ['male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'female']
}
df = pd.DataFrame(data)

# Encode gender to a numeric value for regression
df['gender'] = df['gender'].map({'male': 1, 'female': 0})

# Fit the model
X = df[['beauty', 'gender']]
y = df['score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

# Percent of Variability Explained
r_squared = model.rsquared
print(f"Percent of Variability Explained: {r_squared * 100:.2f}%")

# Plotting
sns.lmplot(x='beauty', y='score', hue='gender', data=df)
plt.title("Beauty vs Evaluation Score by Gender")
plt.xlabel("Beauty")
plt.ylabel("Evaluation Score")
plt.show()

# Equation for male professors
male_intercept = model.params[0] + model.params[1]  # male intercept
male_slope = model.params[2]  # slope for beauty
print(f"Equation for Male Professors: score = {male_intercept:.2f} + {male_slope:.2f} * beauty")
##################################################
#Question 26.0, Round 2 with threat_id: thread_vaz2cBR0ys2fW6wo5W6t8hpc
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data Creation (Replace this with your dataset)
# df = pd.read_csv('your_dataset.csv')
# Assuming the dataframe has 'score', 'bty_avg', and 'gender' columns
# For example, you can replace the DataFrame creation with your actual dataset loading code
data = {
    'score': np.random.uniform(3, 5, 100),  # average professor evaluation score
    'bty_avg': np.random.uniform(1, 10, 100),  # average beauty rating
    'gender': np.random.choice(['Male', 'Female'], 100)  # gender variable
}
df = pd.DataFrame(data)

# Encode gender as binary for regression (0 for Female, 1 for Male)
df['gender_encoded'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Define the independent variables (X) and the dependent variable (y)
X = df[['bty_avg', 'gender_encoded']]
y = df['score']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpretation of the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

# Output the interpretations
print(f'Intercept: {intercept:.2f}, which represents the average score when beauty rating is 0 and gender is Female.')
print(f'Slope for beauty rating: {slope_bty:.2f}, which indicates that for each one unit increase in beauty rating, the average score is expected to increase by {slope_bty:.2f}.')
print(f'Slope for gender: {slope_gender:.2f}, meaning that if the gender changes from Female to Male, the score is expected to increase by {slope_gender:.2f}.')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='bty_avg', y='score', hue='gender', dodge=True, jitter=True, alpha=0.6)
plt.title('Average Professor Evaluation Score by Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 2 with threat_id: thread_TeO3Xrn1Al2J9d7CIW5JcYFo
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
# Assuming 'data.csv' contains 'beauty', 'evaluation_score', and 'gender' columns
data = pd.read_csv('data.csv')

# Create dummy variables for gender
data['gender_male'] = np.where(data['gender'] == 'male', 1, 0)
data['gender_female'] = np.where(data['gender'] == 'female', 1, 0)

# Fit the linear regression model
X = data[['beauty', 'gender_male', 'gender_female']]
Y = data['evaluation_score']
X = sm.add_constant(X)  # Adds constant term to the model

model = sm.OLS(Y, X).fit()

# Summary of the model
print(model.summary())

# Predict scores for male professors
X_male = data[data['gender'] == 'male'][['beauty']]
X_male = sm.add_constant(X_male)  # Adds constant term
predictions_male = model.predict(X_male)

# Predict scores for female professors
X_female = data[data['gender'] == 'female'][['beauty']]
X_female = sm.add_constant(X_female)  # Adds constant term
predictions_female = model.predict(X_female)

# Plotting
plt.scatter(data[data['gender'] == 'male']['beauty'], 
            data[data['gender'] == 'male']['evaluation_score'], 
            color='blue', label='Male Professors')
plt.scatter(data[data['gender'] == 'female']['beauty'], 
            data[data['gender'] == 'female']['evaluation_score'], 
            color='red', label='Female Professors')

plt.plot(data[data['gender'] == 'male']['beauty'], predictions_male, color='blue', linewidth=2)
plt.plot(data[data['gender'] == 'female']['beauty'], predictions_female, color='red', linewidth=2)

plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 3 with threat_id: thread_3QbWBspxCl5zjRJWL110ugK8
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Assuming the data is in a CSV file named "data.csv"
data = pd.read_csv("data.csv")

# Prepare the data
X = pd.DataFrame({
    'bty_avg': data['bty_avg'],       # Average beauty rating
    'gender': pd.get_dummies(data['gender'], drop_first=True)  # One-hot encode gender
})
y = data['score']                    # Average professor evaluation score

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params[1]  # Adjust index if different for gender variable

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, palette='Set2', dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 3 with threat_id: thread_cs0ytNqUBJRmOh17mWOfjQDN
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('your_data.csv')  # replace with your actual data file

# Prepare the data for regression
# Assuming 'score' is the evaluation score, 'beauty' is the beauty score, and 'gender' is the gender
data['gender'] = data['gender'].astype('category')

# Fit the regression model
model = sm.OLS(data['score'], sm.add_constant(data[['beauty', 'gender']])).fit()

# Get the R-squared value (percent of variability explained)
r_squared = model.rsquared * 100
print(f'Percent of variability explained by the model: {r_squared:.2f}%')

# Get the equation for just male professors
male_model = sm.OLS(data[data['gender'] == 'male']['score'], sm.add_constant(data[data['gender'] == 'male']['beauty'])).fit()
male_intercept = male_model.params['const']
male_slope = male_model.params['beauty']
print(f'Equation for male professors: score = {male_intercept:.2f} + {male_slope:.2f} * beauty')

# Relationship between beauty and evaluation score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='beauty', y='score', hue='gender', palette='deep')
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 4 with threat_id: thread_czPINSsD4COCeHVUfBQLNfHT
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame containing the data

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy/indicator variables
Y = df['score']

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(Y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpretation of intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_male']  # Assuming male is the dummy variable

print(f"Intercept (average score when bty_avg is 0 and gender is female): {intercept}")
print(f"Slope for bty_avg (change in score for a one-unit increase in beauty rating): {slope_bty}")
print(f"Slope for gender (change in score moving from female to male): {slope_gender}")

# Create the scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 4 with threat_id: thread_hhtz8bHNLSWD7j6HXnkjoV9B
import pandas as pd
import statsmodels.api as sm

# Sample data frame creation for illustration
# Replace this with your actual dataset
data = {
    'beauty_score': [1, 2, 3, 4, 5],
    'evaluation_score': [2, 3, 4, 5, 6],
    'gender': ['male', 'male', 'female', 'female', 'male']
}
df = pd.DataFrame(data)

# Creating dummy variables for gender
df['gender_male'] = (df['gender'] == 'male').astype(int)

# Define independent variables (including constant)
X = df[['beauty_score', 'gender_male']]
X = sm.add_constant(X)  # Add constant term to the predictor

# Define dependent variable
y = df['evaluation_score']

# Fit model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# To get the equation for male professors
male_coef = model.params['beauty_score']  # Coefficient for beauty score
intercept = model.params['const']  # Intercept
print("Equation for Male Professors: evaluation_score =", intercept, "+", male_coef, "* beauty_score")

# Percentage of variability explained (R-squared)
print("R-squared (percent of variability explained):", model.rsquared)
##################################################
#Question 26.0, Round 5 with threat_id: thread_bJV9iHkwAqdO3z8tfrSDxzSP
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming the data is in a DataFrame named 'data'
# 'score_avg' is the average professor evaluation score
# 'bty_avg' is the average beauty rating
# 'gender' is the gender of the professors

# Generate some example data (this should be replaced with your actual data)
# data = pd.DataFrame({
#     'score_avg': np.random.random(100) * 5,  # Scores between 0 and 5
#     'bty_avg': np.random.random(100) * 5,    # Beauty ratings between 0 and 5
#     'gender': np.random.choice(['Male', 'Female'], size=100)
# })

# Preparing the data for the regression model
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})  # Encoding gender
X = data[['bty_avg', 'gender']]  # Independent variables
y = data['score_avg']  # Dependent variable

# Adding a constant for the intercept term
X = sm.add_constant(X)

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Displaying the model summary
print(score_bty_gender_fit.summary())

# Intercept and slope interpretation
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True, alpha=0.7)
plt.title('Average Score by Beauty Rating (Colored by Gender)')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper right')
plt.show()
##################################################
#Question 26.1, Round 5 with threat_id: thread_hwMIeRxq4r9LCTze8b97z6ZW
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# Example dataset loading (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('your_dataset.csv')  # Assumes the dataset has 'score', 'beauty', and 'gender' columns

# Split data into male and female professors
male_data = data[data['gender'] == 'Male']
female_data = data[data['gender'] == 'Female']

# Fitting the model for both genders
X_male = sm.add_constant(male_data['beauty'])  # Add an intercept
model_male = sm.OLS(male_data['score'], X_male).fit()

X_female = sm.add_constant(female_data['beauty'])  # Add an intercept
model_female = sm.OLS(female_data['score'], X_female).fit()

# Calculate the percent of variability explained
r_squared_male = model_male.rsquared
r_squared_female = model_female.rsquared

print(f'Male Model Equation: score = {model_male.params[0]} + {model_male.params[1]} * beauty')
print(f'Percent Variability Explained by Male Model: {r_squared_male * 100:.2f}%')

print(f'Female Model Equation: score = {model_female.params[0]} + {model_female.params[1]} * beauty')
print(f'Percent Variability Explained by Female Model: {r_squared_female * 100:.2f}%')

# Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(male_data['beauty'], male_data['score'], color='blue', label='Male Professors')
plt.scatter(female_data['beauty'], female_data['score'], color='red', label='Female Professors')

# Plot regression lines
plt.plot(male_data['beauty'], model_male.predict(X_male), color='blue', linewidth=2)
plt.plot(female_data['beauty'], model_female.predict(X_female), color='red', linewidth=2)

plt.title('Evaluation Score vs Beauty by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 6 with threat_id: thread_R48EybAifjAnNK7zywyeA7vy
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data creation - replace this with your actual data loading
# Assuming 'data' is a DataFrame with columns ['score', 'bty_avg', 'gender']
# data = pd.read_csv('your_data.csv')  # Load your data here

# Example DataFrame for demonstration
data = pd.DataFrame({
    'score': [3.5, 4.1, 3.9, 3.0, 4.0],
    'bty_avg': [2, 5, 3, 1, 4],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Female']
})

# Encode gender as a binary variable (0 for Male, 1 for Female)
data['gender_encoded'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Define independent variables and dependent variable
X = data[['bty_avg', 'gender_encoded']]
y = data['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create jittered scatter plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, palette='Set2')
plt.title('Average Score by Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 6 with threat_id: thread_tNoo5uINRNc7JscPOrzTx1wW
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Divide the data into male and female
male_professors = df[df['gender'] == 'male']
female_professors = df[df['gender'] == 'female']

# Fit model for males
X_male = sm.add_constant(male_professors['beauty'])
model_male = sm.OLS(male_professors['evaluation_score'], X_male).fit()

# Fit model for females
X_female = sm.add_constant(female_professors['beauty'])
model_female = sm.OLS(female_professors['evaluation_score'], X_female).fit()

# Print the summaries
print("Male Professors Model Summary:")
print(model_male.summary())
print("\nFemale Professors Model Summary:")
print(model_female.summary())

# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='beauty', y='evaluation_score', hue='gender', alpha=0.6)
plt.plot(male_professors['beauty'], model_male.predict(X_male), color='blue', label='Male Fit')
plt.plot(female_professors['beauty'], model_female.predict(X_female), color='orange', label='Female Fit')
plt.title("Evaluation Score vs. Beauty")
plt.xlabel("Beauty Score")
plt.ylabel("Evaluation Score")
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 7 with threat_id: thread_4rlVdYILJ6W4UB6LGpzdg1mB
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Replace 'your_data.csv' with the path to your dataset
data = pd.read_csv('your_data.csv')

# Prepare the data
# Assume 'score_avg' is the average score, 'bty_avg' is the average beauty rating, and 'gender' is the gender of the professors.
# Convert gender to a numeric variable (0 for male, 1 for female), if necessary
data['gender_numeric'] = data['gender'].map({'male': 0, 'female': 1})

# Define the independent variables and dependent variable
X = data[['bty_avg', 'gender_numeric']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = data['score_avg']

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty_avg = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_numeric']

print(f"Intercept: {intercept}")
print(f"Slope (Beauty Rating): {slope_bty_avg}")
print(f"Slope (Gender): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True, dodge=True, alpha=0.6)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 8 with threat_id: thread_F75hsUw0emoyEwsGWqaFpVLi
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have your data in a DataFrame named 'df'
# Here 'bty_avg' is average beauty rating, 'gender' is a binary variable (0 for male, 1 for female), and 'score' is the evaluation score.

# Example DataFrame creation (replace this with your actual DataFrame)
# df = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = sm.add_constant(X)  # Adds the intercept
y = df['score']
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Intercept interpretation
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create a scatterplot of score by bty_avg with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='best')
plt.grid()
plt.show()
##################################################
#Question 26.0, Round 9 with threat_id: thread_C8tR1ubAIlzhB33ktJVt0rIc
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv('your_data_file.csv')  # Ensure to specify your actual data file path

# Encode gender as a numerical variable (assuming 'Male' = 1 and 'Female' = 0)
data['gender_encoded'] = data['gender'].map({'Male': 1, 'Female': 0})

# Define the independent variables and dependent variable
X = data[['bty_avg', 'gender_encoded']]
y = data['professor_evaluation_score']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='professor_evaluation_score', hue='gender', data=data, jitter=True, alpha=0.6)
plt.title("Scatterplot of Professor Evaluation Score by Beauty Rating")
plt.xlabel("Average Beauty Rating")
plt.ylabel("Average Professor Evaluation Score")
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 9 with threat_id: thread_pSgMoZrLnvszlcL2kgXIMog7
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (assuming you have a DataFrame with columns "beauty", "evaluation_score", "gender")
data = {
    'beauty': [5, 6, 7, 8, 9, 4, 5, 6, 3, 8],  # Example beauty scores
    'evaluation_score': [3, 4, 5, 6, 7, 3, 4, 6, 2, 5],  # Example evaluation scores
    'gender': ['male', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
}

df = pd.DataFrame(data)

# Fit linear regression model
X = df[['beauty']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['evaluation_score']

# Fit model for males
male_df = df[df['gender'] == 'male']
X_male = male_df[['beauty']]
X_male = sm.add_constant(X_male)
y_male = male_df['evaluation_score']
male_model = sm.OLS(y_male, X_male).fit()

# Fit model for females
female_df = df[df['gender'] == 'female']
X_female = female_df[['beauty']]
X_female = sm.add_constant(X_female)
y_female = female_df['evaluation_score']
female_model = sm.OLS(y_female, X_female).fit()

# Output results
print(f'Male model summary:\n{male_model.summary()}')
print(f'Female model summary:\n{female_model.summary()}')
print(f'Male equation of the line: y = {male_model.params[0]} + {male_model.params[1]} * beauty')
print(f'Female equation of the line: y = {female_model.params[0]} + {female_model.params[1]} * beauty')

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='beauty', y='evaluation_score', hue='gender')
plt.plot(male_df['beauty'], male_model.predict(X_male), color='blue', label='Male Fit', linewidth=2)
plt.plot(female_df['beauty'], female_model.predict(X_female), color='orange', label='Female Fit', linewidth=2)
plt.title('Beauty vs Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()

# Explained variance
print('Male R-squared:', male_model.rsquared)
print('Female R-squared:', female_model.rsquared)
##################################################
#Question 26.0, Round 10 with threat_id: thread_AYCUq05vmXayMl5daVos3X9t
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your data
# df = pd.read_csv('your_data.csv')  # Uncomment to load your data

# Example DataFrame structure (you need to replace this with your actual data)
# df = pd.DataFrame({
#     'score': [3.5, 4.2, 4.0, 3.8, 4.5],
#     'bty_avg': [4.0, 5.0, 4.5, 3.5, 5.0],
#     'gender': ['M', 'F', 'M', 'F', 'M']
# })

# Convert gender to numerical values (e.g., 0 for Male, 1 for Female)
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# Define the dependent variable and independent variables
X = df[['bty_avg', 'gender']]  # Independent variables
y = df['score']  # Dependent variable

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, alpha=0.6)
plt.title('Scatterplot of Score by Average Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
plt.grid()
plt.show()
##################################################
#Question 26.1, Round 10 with threat_id: thread_dWngA3kcf349okhu90QOi9Q3
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (ensure you have your CSV file or data frame ready)
# Example: df = pd.read_csv('your_data.csv')
df = pd.DataFrame() # Replace this with your actual DataFrame

# Fit the model
X = df[['beauty_score', 'gender']]  # Assuming you have these columns
X = pd.get_dummies(X, drop_first=True)  # Convert categorical gender to dummy variables
y = df['evaluation_score']  # Replace with your actual evaluation score column

X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()  # Fit the model

# Print model summary to get the R-squared value and coefficients
print(model.summary())

# Get coefficients for male and female
coef = model.params
slope_male = coef['beauty_score'] + coef['beauty_score:gender_male']  # Adjust based on your dummies 
intercept_male = coef['const']

# Output the equation for male professors
print(f"Equation for male professors: y = {slope_male}x + {intercept_male}")

# Visualizing relationship between beauty and score
sns.lmplot(data=df, x='beauty_score', y='evaluation_score', hue='gender',
           palette={"male": "blue", "female": "red"}, markers=["o", "s"])

plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()

# Variability explained by the model
r_squared = model.rsquared
print(f'Variability explained by the model: {r_squared * 100:.2f}%')
##################################################
#Question 26.0, Round 11 with threat_id: thread_Z6SGcQ8RDzfCsJ6N0W8KVisQ
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Replace 'your_data.csv' with the path to your actual data file
# data = pd.read_csv('your_data.csv')

# Example dataframe structure: 
# data = pd.DataFrame({
#     'bty_avg': [4.2, 3.5, 5.0, 4.0, ...],
#     'gender': ['Male', 'Female', 'Female', 'Male', ...],
#     'score': [3.8, 4.2, 5.0, 4.5, ...]
# })

# Fit the multiple linear regression model
data['gender'] = pd.Categorical(data['gender'])
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable to dummy variables
y = data['score']
X = sm.add_constant(X)  # Add a constant term for the intercept

score_bty_gender_fit = sm.OLS(y, X).fit()

# Output the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty_avg = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Replace with actual dummy name if necessary

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty_avg}")
print(f"Slope for gender (Male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, palette='Set1')
plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 11 with threat_id: thread_1YoMZpbNuNob6TYwZkJbzux0
import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (modify the path and filename as needed)
# df = pd.read_csv('your_data.csv')

# Replace the following line with your own data for the regression analysis
# df = pd.DataFrame({'score': [/*scores*/], 'beauty': [/*beauty scores*/], 'gender': [/*'male' or 'female'*/]})

# Fit the model accounting for beauty and gender
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable (gender) into dummy/indicator variables
y = df['score']

X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

# Analyze the percentage of variability explained by the model (R-squared)
r_squared = model.rsquared
percent_explained = r_squared * 100
print(f"The model explains {percent_explained:.2f}% of the variability in scores.")

# Analyze relationship for just male professors
male_professors = df[df['gender'] == 'male']
X_male = male_professors['beauty']
y_male = male_professors['score']
coefficients = model.params

# Equation of the line for male professors
slope = coefficients['beauty']  # Coefficient for beauty
intercept = coefficients['const']  # Intercept
print(f"The equation of the line for male professors is: score = {intercept:.2f} + {slope:.2f} * beauty")

# Visualize the relationship
sns.scatterplot(data=df, x='beauty', y='score', hue='gender', palette='viridis')
plt.plot(X_male['beauty'], intercept + slope * X_male['beauty'], color='blue')  # Line for male professors
plt.title("Beauty vs Evaluation Score by Gender")
plt.xlabel("Beauty")
plt.ylabel("Evaluation Score")
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 12 with threat_id: thread_4mKHeIyzMsAMXwfdKfK4wYBv
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Assuming you have a DataFrame named 'df' with columns 'score_avg', 'bty_avg', and 'gender'
# df = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy variables
y = df['score_avg']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Model results
print(model.summary())

# Interpret the coefficients
intercept = model.params['const']
slope_bty = model.params['bty_avg']  # Assuming 'bty_avg' is the column for beauty ratings
slope_gender = model.params['gender_Male']  # Assuming 'gender_Male' is the dummy variable for male

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender (vs Female): {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, alpha=0.7)
plt.title('Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 12 with threat_id: thread_Fqiy4wUOKQAqjEkXFZFy3Pc9
import pandas as pd
import statsmodels.api as sm

# Assuming you have a DataFrame 'data' with columns 'beauty', 'score', and 'gender'
# Load your data
data = pd.read_csv('your_data_file.csv')  # Update with your actual data file

# Fit model
model = sm.OLS(data['score'], sm.add_constant(data[['beauty', 'gender']])).fit()

# 1. Percent of Variability Explained
r_squared = model.rsquared
print(f"Percent of variability explained: {r_squared * 100:.2f}%")

# 2. Equation of the line for male professors
male_data = data[data['gender'] == 'male']
male_model = sm.OLS(male_data['score'], sm.add_constant(male_data['beauty'])).fit()
print(f"Equation for male professors: Score = {male_model.params[0]:.2f} + {male_model.params[1]:.2f} * Beauty")

# 3. Relationship Variation
female_data = data[data['gender'] == 'female']
female_model = sm.OLS(female_data['score'], sm.add_constant(female_data['beauty'])).fit()

# Compare coefficients
print(f"Male coefficient: {male_model.params[1]:.2f}, Female coefficient: {female_model.params[1]:.2f}")
##################################################
#Question 26.0, Round 13 with threat_id: thread_IzNT92xHpNKocMY5lHKbQP5E
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
# Assuming df is your DataFrame containing the data.
# df = pd.read_csv('your_data_file.csv')  # Uncomment and modify this line to load your dataset.

# For demonstration, we will create a sample DataFrame (replace this with your data).
data = {
    'score_avg': np.random.uniform(2, 5, 100),
    'bty_avg': np.random.uniform(1, 10, 100),
    'gender': np.random.choice(['Male', 'Female'], 100)
}
df = pd.DataFrame(data)

# Encode gender as a categorical variable (0 for Male, 1 for Female)
df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})

# Define the independent variables and dependent variable
X = df[['bty_avg', 'gender_encoded']]
y = df['score_avg']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpretation of the model
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f"Intercept: {intercept}")
print(f"Slope (Beauty Rating): {slope_bty}")
print(f"Slope (Gender): {slope_gender}")

# Plot the scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, palette='Set1', dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Scores by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 14 with threat_id: thread_vrJFet0Du5GeGtIk58oTs1fU
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data generation (remove this part when using actual data)
# df = pd.DataFrame({
#     'bty_avg': np.random.rand(100) * 10,  # Average beauty rating between 0 and 10
#     'gender': np.random.choice(['Male', 'Female'], 100),  # Randomly assign genders
#     'score': np.random.rand(100) * 5 + 2.5  # Average professor evaluation score between 2.5 and 7.5
# })

# Load your actual dataset
# df = pd.read_csv('your_file.csv')  # Uncomment this line to read your actual data

# Encode gender into numerical format for regression
df = pd.get_dummies(df, columns=['gender'], drop_first=True)  # This will encode 'gender_Male'

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender_Male']]  # Use encoded gender
y = df['score']
X = sm.add_constant(X)  # Add constant for intercept
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params[0]
bty_slope = score_bty_gender_fit.params[1]
gender_slope = score_bty_gender_fit.params[2]
print("Intercept:", intercept)
print("Slope of beauty rating (bty_avg):", bty_slope)
print("Slope of gender (gender_Male):", gender_slope)

# Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bty_avg', y='score', hue='gender', jitter=True)
plt.title('Average Professor Evaluation Score vs. Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 14 with threat_id: thread_KObGdLB9hKZSVdC3tFEnNFzO
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data_file.csv')

# Example data
# df = pd.DataFrame({
#     'beauty': np.random.rand(100),
#     'evaluation_score': np.random.rand(100) * 10,
#     'gender': np.random.choice(['male', 'female'], size=100)
# })

# Create dummy variables for gender
df['male'] = np.where(df['gender'] == 'male', 1, 0)
df['female'] = np.where(df['gender'] == 'female', 1, 0)

# Define independent variables and the dependent variable
X = df[['beauty', 'male']]
y = df['evaluation_score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

# Percentage of variability explained (R-squared)
percent_variability_explained = model.rsquared * 100
print(f"Percent of variability explained: {percent_variability_explained:.2f}%")

# Equation for male professors (the coefficients)
intercept = model.params['const']
coef_beauty = model.params['beauty']
equation_male = f"evaluation_score = {intercept:.2f} + {coef_beauty:.2f} * beauty"

print(f"Equation for male professors: {equation_male}")

# Visualize the relationship
sns.lmplot(x='beauty', y='evaluation_score', hue='gender', data=df, markers=["o", "x"])
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 15 with threat_id: thread_AK51D6l8Tzc0ZE8o3fPCwQv7
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace 'your_data.csv' with the actual path to your dataset
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X['gender'] = X['gender'].map({'male': 0, 'female': 1})  # Convert gender to numerical (0 for male, 1 for female)
y = data['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Output the model summary
print(score_bty_gender_fit.summary())

# Interpretation
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope of beauty rating: {slope_beauty}')
print(f'Slope of gender (female): {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Scatterplot of Score by Average Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 16 with threat_id: thread_UjRsYkfqgqyOkLeTpkCUn7wV
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your dataset is in a DataFrame called 'df' with columns: 'score', 'bty_avg', and 'gender'
# Example: df = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy/indicator variables
Y = df['score']

X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the regression model
score_bty_gender_fit = sm.OLS(Y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, alpha=0.6, dodge=True)
plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 16 with threat_id: thread_dhqqjGz3rtKMwVaJ90xjvnq6
import pandas as pd
import statsmodels.api as sm

# Assuming 'data' is a DataFrame that contains your dataset with 'score', 'beauty', and 'gender' columns.

# Step 1: Prepare the data for modeling
data['gender'] = data['gender'].astype('category')

# Step 2: Fit the linear regression model
X = sm.add_constant(data[['beauty', 'gender']])  # Adding constant for intercept
y = data['score']
model = sm.OLS(y, X).fit()

# Step 3: Print the summary statistics which includes the R-squared
print(model.summary())

# Step 4: Extracting and formatting relevant information
r_squared = model.rsquared * 100  # Percent
male_coef = model.params['beauty']  # Assuming 'beauty' is the variable for beauty
intercept = model.params['const']
print(f"R-squared Value: {r_squared}%")
print(f"Equation for Male Professors: Score = {intercept} + {male_coef} * Beauty")

# Step 5: Comparing coefficients between genders
coefficients = model.params
print("Coefficients:")
print(coefficients)
##################################################
#Question 26.0, Round 17 with threat_id: thread_eEm0f1h7ETvNS3EUIMqHkdVJ
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data generation (replace this section with loading your actual dataset)
data = {
    'score': [4.5, 4.0, 4.2, 3.9, 4.8, 5.0, 3.5, 4.1, 4.4, 4.7],
    'bty_avg': [7, 6, 5, 4, 8, 9, 3, 6, 7, 8],
    'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
}
df = pd.DataFrame(data)

# Encoding gender as a binary variable
df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender_encoded']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['score']
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Coefficients Interpretation
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Scatterplot with jitter
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Scatterplot of Score by Beauty Average with Jitter')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 17 with threat_id: thread_jeOZMqBg6SDZ7a940N7mdrnn
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Ensure to replace 'your_data.csv' with the actual file name or path if needed.
data = pd.read_csv('your_data.csv')

# Assuming 'score' is the evaluation score, 'beauty' is the beauty score, and 'gender' is a binary variable (0 for male, 1 for female)

# Fit the linear regression model
X = data[['beauty', 'gender']]
X = sm.add_constant(X)  # Adding a constant term
y = data['score']

model = sm.OLS(y, X).fit()

# Percent of variability explained (R-squared)
r_squared = model.rsquared * 100  # To percentage

# Output the R-squared value
print(f"Percent of variability in score explained by the model: {r_squared:.2f}%")

# Equation of the line for male professors (gender == 0)
coefficients = model.params
slope = coefficients['beauty']
intercept = coefficients['const']
equation_male = f"Score = {intercept:.2f} + {slope:.2f} * Beauty"

# Print the equation for male professors
print(f"Equation of the line for male professors: {equation_male}")

# To visualize the interaction between beauty and evaluation score for both genders:
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='beauty', y='score', hue='gender', palette=['blue', 'pink'], alpha=0.6)
plt.title("Beauty vs Evaluation Score by Gender")
plt.xlabel("Beauty Score")
plt.ylabel("Evaluation Score")

# Predicting values for males and females
predicted_scores = model.predict(X)
data['predicted_score'] = predicted_scores

# Plotting the fitted lines for males and females
sns.regplot(data=data[data['gender'] == 0], x='beauty', y='predicted_score', scatter=False, color='blue', label='Male Fit', line_kws={'linestyle':'--'})
sns.regplot(data=data[data['gender'] == 1], x='beauty', y='predicted_score', scatter=False, color='pink', label='Female Fit', line_kws={'linestyle':'--'})

plt.legend()
plt.show()
##################################################
#Question 26.0, Round 18 with threat_id: thread_Leex9naunJxpA5DL7bGEy4eZ
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with your actual data file)
data = pd.read_csv('your_data.csv')

# Assuming the columns are named 'score', 'bty_avg', and 'gender'
# Prepare the data
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
y = data['score']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Change 'gender_Male' to the appropriate dummy variable name

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Create a jittered scatterplot of score by average beauty rating and color by gender
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, alpha=0.6)
plt.title('Scatterplot of Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 18 with threat_id: thread_eGao09mTmH6SKZ3khv4PIbqq
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data - replace with your actual DataFrame
# df = pd.read_csv("your_data.csv")

# Assuming df has 'evaluation_score', 'beauty_score', and 'gender' columns
# Fit the model for both genders
model_fit = sm.OLS.from_formula('evaluation_score ~ beauty_score * gender', data=df).fit()

# Output the summary of the model
print(model_fit.summary())

# Calculate the % of variability in score explained by the model (R-squared)
r_squared = model_fit.rsquared
print(f"Percent of variability explained by the model: {r_squared * 100:.2f}%")

# Create separate plots for male and female professors
for gender in df['gender'].unique():
    subset = df[df['gender'] == gender]
    sns.regplot(x='beauty_score', y='evaluation_score', data=subset, label=gender)
    
plt.title('Evaluation Score vs Beauty Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 19 with threat_id: thread_A3X6y6im4PW2oTcE3cUGcJ8B
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Replace 'your_data.csv' with the actual file name
df = pd.read_csv('your_data.csv')

# Assume columns for average professor evaluation score, average beauty rating, gender
# Rename columns as necessary to match your DataFrame
df.rename(columns={'average_eval': 'score', 'average_beauty': 'bty_avg', 'gender': 'gender'}, inplace=True)

# Jitter the data for better visualization
df['bty_avg_jittered'] = df['bty_avg'] + np.random.normal(0, 0.05, size=len(df))

# Fit the multiple linear regression model
X = pd.get_dummies(df[['bty_avg', 'gender']], drop_first=True)  # Convert gender to dummy variables
X = sm.add_constant(X)  # Adds a constant term for the intercept
y = df['score']
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
# Assuming 'gender_Male' is the column created for male gender
slope_gender = score_bty_gender_fit.params['gender_Male']  # Change based on your dummy variable naming

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender (Male versus Female): {slope_gender}')

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bty_avg_jittered', y='score', hue='gender', alpha=0.6)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating (Jittered)')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.grid(True)
plt.show()
##################################################
#Question 26.1, Round 19 with threat_id: thread_inxm314zUmAx6CHtuGNGMSpg
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('your_data_file.csv')  # replace with your actual data file

# Fit the model (assuming the model has beauty score and gender in the dataset)
# Replace 'Score', 'Beauty', 'Gender' with your actual column names
X = df[['Beauty', 'Gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical gender to dummy variables
y = df['Score']

# Add a constant for the intercept
X = sm.add_constant(X) 

# Fit the model
model = sm.OLS(y, X).fit()

# Percent of variability explained by the model
r_squared = model.rsquared
print(f"R-squared: {r_squared * 100:.2f}% of the variability in score is explained by the model.")

# Equation of the line for male professors (assuming male is the first category)
if 'Gender_male' in X.columns:
    male_coefficients = model.params[['const', 'Beauty', 'Gender_male']]
    print("Equation for male professors: Score =", male_coefficients['const'], "+", male_coefficients['Beauty'], "* Beauty +", male_coefficients['Gender_male'], "* Gender")

# Visualize the relationship
sns.lmplot(x='Beauty', y='Score', hue='Gender', data=df, aspect=1.5)
plt.title("Beauty vs Evaluation Score")
plt.xlabel("Beauty Score")
plt.ylabel("Evaluation Score")
plt.show()
##################################################
#Question 26.0, Round 20 with threat_id: thread_M7GIg0pU2xgfvI7RmFuR5F6S
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the data is in a CSV file named 'data.csv'
# Load the dataset
data = pd.read_csv('data.csv')

# Define the independent variables and the dependent variable
X = data[['bty_avg', 'gender']]
y = data['score']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Fit the multiple linear regression model
X = sm.add_constant(X)  # Add a constant term to the predictor
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and the slopes
intercept = score_bty_gender_fit.params[0]
slope_bty_avg = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope for Beauty Rating: {slope_bty_avg}")
print(f"Slope for Gender: {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='bty_avg', y='score', hue='gender', jitter=True)
plt.title('Scatterplot of Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 21 with threat_id: thread_rYhgGGEl83kLcfJ92kSZdHKu
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your dataset
# Assuming your CSV file is named 'data.csv' and has columns 'score', 'bty_avg', and 'gender'
# Replace 'data.csv' with the path of your file
df = pd.read_csv('data.csv')

# Prepare the data for the regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to numerical
y = df['score']

# Fit the multiple linear regression model
model = sm.OLS(y, sm.add_constant(X)).fit()
score_bty_gender_fit = model

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret coefficients
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Create a jittered scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 21 with threat_id: thread_22yDwCWmoEb2jNLZFKYAMB1I
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Ensure your CSV file has columns named 'beauty', 'score', and 'gender'
data = pd.read_csv('your_data_file.csv')

# Create a boolean mask for male professors
male_professors = data[data['gender'] == 'male']
female_professors = data[data['gender'] == 'female']

# Fit the model
X_male = sm.add_constant(male_professors['beauty'])  # Add constant for intercept
model_male = sm.OLS(male_professors['score'], X_male).fit()

# Summary of the model for male professors
print(model_male.summary())

# Extract equation parameters
intercept_male = model_male.params[0]
slope_male = model_male.params[1]
print(f'The equation of the line for male professors: score = {intercept_male:.2f} + {slope_male:.2f} * beauty')

# Fit the model for female professors
X_female = sm.add_constant(female_professors['beauty'])
model_female = sm.OLS(female_professors['score'], X_female).fit()

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='beauty', y='score', hue='gender', data=data)
plt.plot(male_professors['beauty'], model_male.predict(X_male), color='blue', label='Male Fit')
plt.plot(female_professors['beauty'], model_female.predict(X_female), color='red', label='Female Fit')
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()

# Explained variability
r_squared_male = model_male.rsquared
print(f'Percent of variability in score explained by the model for male professors: {r_squared_male * 100:.2f}%')
##################################################
#Question 26.1, Round 22 with threat_id: thread_2MztM77QnOTFobVxt8KAAHi4
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data (replace 'your_data.csv' with the actual filename)
data = pd.read_csv('your_data.csv')

# Assuming the relevant columns are named 'score', 'bty_avg', and 'gender'
# We will check the first few rows of the data
print(data.head())

# Prepare the data for the regression model
data = data.dropna(subset=['score', 'bty_avg', 'gender'])  # Drop rows with missing values
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})  # Encode gender (if needed)

# Define the dependent and independent variables
X = data[['bty_avg', 'gender']]
y = data['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender (Male): {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, alpha=0.6)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()
##################################################
#Question 26.0, Round 22 with threat_id: thread_rVxYGXQ2wVe7Xhybh5SAs884
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment and update this line with your actual dataset path

# Example of a sample dataset format
data = {
    'score': [4, 3, 5, 4, 3],
    'beauty': [7, 5, 6, 8, 4],
    'gender': ['male', 'female', 'male', 'female', 'male']
}
df = pd.DataFrame(data)

# Convert gender to a categorical variable
df['gender'] = df['gender'].astype('category')

# Fit the model
model = sm.OLS.from_formula('score ~ beauty * gender', data=df).fit()

# Variability explained by the model
r_squared = model.rsquared * 100  # To get percentage
print(f"Percentage of variability explained by the model: {r_squared:.2f}%")

# Get the equation for male professors
male_coef = model.params['Intercept'] + model.params['beauty'] * df.loc[df['gender'] == 'male', 'beauty'].mean()
male_equation = f"score = {model.params['Intercept']:.2f} + {model.params['beauty']:.2f}*beauty + {model.params['gender[T.male]']:.2f}"
print(f"Equation for male professors: {male_equation}")

# Plotting
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='beauty', y='score', hue='gender', style='gender')
plt.title('Scatter plot of Beauty vs Evaluation Score')
plt.xlabel('Beauty Rating')
plt.ylabel('Evaluation Score')
plt.legend(title='Gender')
plt.show()

# Analysis of how the relationship varies
sns.lmplot(data=df, x='beauty', y='score', hue='gender', aspect=1.5)
plt.title('Regression lines for Male and Female')
plt.show()
##################################################
#Question 26.1, Round 23 with threat_id: thread_HJ8HRFHUF0CuNW74aGVnVYTq
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into a DataFrame (replace 'your_data.csv' with your actual data file)
# df = pd.read_csv('your_data.csv')  # Uncomment and modify to load your data
# Example DataFrame structure
# df = pd.DataFrame({
#     'score_avg': [3.5, 4.2, 4.0, 2.9, 3.8],
#     'bty_avg': [3.0, 4.0, 3.5, 2.0, 4.5],
#     'gender': ['male', 'female', 'female', 'male', 'female']
# })

# Ensure 'gender' is converted to a numerical format if it's categorical
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
y = df['score_avg']
X = sm.add_constant(X)  # Adds the intercept
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f'Intercept: {intercept:.4f} - This is the expected average professor evaluation score when beauty rating is 0 and gender is male.')
print(f'Slope for beauty rating: {slope_bty:.4f} - This indicates the change in the average professor evaluation score for each unit increase in beauty rating.')
print(f'Slope for gender: {slope_gender:.4f} - This indicates the change in the average professor evaluation score when the gender changes from male to female.')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, palette='Set1')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.0, Round 23 with threat_id: thread_vijCwPq2qtcWmqB1cGNkWeSl
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Assume columns are named 'beauty', 'score', 'gender'
# Creating a dummy variable for gender
df['male'] = np.where(df['gender'] == 'male', 1, 0)

# Fit the regression model
X = df[['beauty', 'male']]
X = sm.add_constant(X)  # Adding a constant term for the intercept
y = df['score']

model = sm.OLS(y, X).fit()

# Get model summary
print(model.summary())

# To extract R-squared value (percent of variability explained)
r_squared = model.rsquared
print(f"Percent of variability explained: {r_squared * 100:.2f}%")

# Coefficients for male professors
male_coeff = model.params['beauty'] + model.params['male'] * model.params['beauty']
intercept_male = model.params['const']
print(f"Equation for male professors: score = {intercept_male:.2f} + {male_coeff:.2f} * beauty")

# Plotting the lines for male and female
plt.scatter(df['beauty'], df['score'], c=df['male'], cmap='coolwarm', alpha=0.5)
x = np.linspace(df['beauty'].min(), df['beauty'].max(), 100)
plt.plot(x, intercept_male + model.params['beauty'] * x, label='Female professor', color='blue')
plt.plot(x, intercept_male + (model.params['beauty'] + model.params['male'] * model.params['beauty']) * x, label='Male professor', color='red')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.title('Beauty vs Evaluation Score by Gender')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 24 with threat_id: thread_Pn1YfngXC9w6jJLYhG0vI8er
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment this line and replace with your dataset

# Example data (you should replace this with your actual data)
data = {
    'score_avg': np.random.uniform(1, 5, 100),  # mock data for average professor evaluation scores
    'bty_avg': np.random.uniform(1, 5, 100),    # mock data for average beauty ratings
    'gender': np.random.choice(['Male', 'Female'], 100) # mock data for gender
}

df = pd.DataFrame(data)

# Convert gender to numerical values (0 for Male, 1 for Female)
df['gender_num'] = df['gender'].map({'Male': 0, 'Female': 1})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender_num']]
X = sm.add_constant(X)  # Adds the intercept term
y = df['score_avg']
model = sm.OLS(y, X).fit()
score_bty_gender_fit = model

# Print the summary of the regression model for interpretation
print(score_bty_gender_fit.summary())

# Interpret coefficients
intercept = model.params['const']
slope_bty = model.params['bty_avg']
slope_gender = model.params['gender_num']

# Interpretation
print(f'Intercept: {intercept}')
print(f'Slope of beauty rating: {slope_bty}')
print(f'Slope of gender (Female vs Male): {slope_gender}')

# Create a jitter scatterplot of score by bty_avg colored by gender
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bty_avg', y='score_avg', hue='gender', jitter=True, alpha=0.7)
plt.title('Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 24 with threat_id: thread_XW8aqTltNYYszndexthEGRej
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load your data
# data = pd.read_csv("your_data.csv") # Make sure your data is formatted correctly

# Example structure (replace with your actual data)
# Assuming 'beauty' is a feature and 'score' is the target variable, and 'gender' is binary (0 for female, 1 for male)
# data = pd.DataFrame({
#     'beauty': np.random.rand(100),
#     'score': np.random.rand(100) * 10,
#     'gender': np.random.choice([0, 1], 100)
# })

# Fit a linear regression model
X = data[['beauty', 'gender']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = data['score']

model = sm.OLS(y, X).fit()

# Outputs
print("Model Summary:")
print(model.summary())

# 1. Percent of variability explained by the model (R-squared)
r_squared = model.rsquared
print(f"Percent of variability in score explained by the model: {r_squared * 100:.2f}%")

# 2. Equation of the line for male professors only (gender = 1)
coefficients = model.params
intercept = coefficients['const']
slope_beauty = coefficients['beauty']
slope_gender = coefficients['gender']
equation_male = f"score = {intercept + slope_gender} + {slope_beauty} * beauty"
print("Equation of the line for male professors:", equation_male)

# 3. The relationship between beauty and evaluation score
# Separate models or coefficients based on gender
model_male = sm.OLS(y[data['gender'] == 1], X[data['gender'] == 1]).fit()
model_female = sm.OLS(y[data['gender'] == 0], X[data['gender'] == 0]).fit()

# Print out the summaries to compare coefficients
print("Male Model Summary:")
print(model_male.summary())
print("Female Model Summary:")
print(model_female.summary())
##################################################
#Question 26.1, Round 25 with threat_id: thread_H2kD8UhF7b8TH37v3VBT9e3F
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
# Make sure to replace 'your_file.csv' with the path to your actual dataset
data = pd.read_csv('your_file.csv')

# Fit the multiple linear regression model
# Assuming 'avg_prof_eval' is the professor evaluation score,
# 'avg_beauty_rating' is the average beauty rating,
# and 'gender' is a binary or categorical variable for gender
X = data[['avg_beauty_rating', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variables
y = data['avg_prof_eval']

X = sm.add_constant(X)  # Add an intercept
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['avg_beauty_rating']
slope_gender = score_bty_gender_fit.params['gender_M']  # assuming 'M' is coded as 1 for male

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_beauty}')
print(f'Slope for Gender (Male): {slope_gender}')

# Create the scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='avg_beauty_rating', y='avg_prof_eval', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 25 with threat_id: thread_sXYGeYIFNlXYuy2F45dINQN3
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame creation, replace this with your actual DataFrame
# Assuming you have a DataFrame 'df' with 'beauty', 'score', and 'gender' columns
df = pd.DataFrame({
    'beauty': np.random.rand(100),
    'score': np.random.rand(100) * 100,
    'gender': np.random.choice(['Male', 'Female'], 100)
})

# Create dummy variables for the gender
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Fit the model
X = df[['beauty', 'gender_Male']]
y = df['score']
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(y, X).fit()
print(model.summary())

# Print the percentage of variability explained
r_squared = model.rsquared
print(f"Percentage of variability explained by the model: {r_squared * 100}%")

# Equation for male professors (with intercept and beauty coefficient)
intercept = model.params['const']
beauty_coef = model.params['beauty']
print(f"Equation for Male Professors: score = {intercept} + {beauty_coef} * beauty")

# For Female professors, since they are the baseline in get_dummies ('gender_Male'), the equation is simply:
# score = intercept (baseline value as beauty impact is 0)

# Plotting the relationship
plt.figure(figsize=(10, 6))
for gender, group in df.groupby('gender'):
    plt.scatter(group['beauty'], group['score'], label=gender)

# Regression lines for both genders
x_values = np.linspace(0, 1, 100)  # assuming beauty score range from 0 to 1
male_scores = intercept + beauty_coef * x_values
female_scores = intercept  # female baseline

plt.plot(x_values, male_scores, color='blue', label='Male Professors', linewidth=2)
plt.plot(x_values, female_scores, color='red', label='Female Professors', linestyle='dashed', linewidth=2)
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Evaluation Score vs. Beauty Score by Gender')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 26 with threat_id: thread_8dyMVmpKa5F6DteI2Jx4RwqH
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
# You can use pd.read_csv() or any other method to load your data
# For example, df = pd.read_csv('your_data.csv')
# Make sure the data contains 'score', 'bty_avg', and 'gender' columns.

# Create a sample DataFrame for demonstration
# Replace this with your actual dataset
data = {
    'score': np.random.rand(100) * 5,  # average professor evaluation score
    'bty_avg': np.random.rand(100) * 5,  # average beauty rating
    'gender': np.random.choice(['Male', 'Female'], size=100)  # gender
}
df = pd.DataFrame(data)

# Convert categorical variable 'gender' to numerical
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = sm.add_constant(X)  # Adds the intercept term
y = df['score']

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params[0]
slope_bty_avg = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope of bty_avg: {slope_bty_avg}")
print(f"Slope of gender: {slope_gender}")

# Create scatterplot of score vs bty_avg colored by gender
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bty_avg', y='score', hue='gender', data=df, alpha=0.6, jitter=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating (bty_avg)')
plt.ylabel('Average Professor Evaluation Score (score)')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 26 with threat_id: thread_duC3wutnTshb31hquEd4I0v5
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv('your_data.csv')  # Replace with your actual data file

# Prepare data for regression, assuming the dataset has 'beauty', 'score', and 'gender' columns
# Create two datasets: one for male and one for female professors
data_male = data[data['gender'] == 'male']
data_female = data[data['gender'] == 'female']

# Fit the model for both male and female professors
X_male = sm.add_constant(data_male['beauty'])  # Add constant for intercept
model_male = sm.OLS(data_male['score'], X_male).fit()
print("Male Professors Model Summary:\n", model_male.summary())

X_female = sm.add_constant(data_female['beauty'])  # Add constant for intercept
model_female = sm.OLS(data_female['score'], X_female).fit()
print("Female Professors Model Summary:\n", model_female.summary())

# Predict and plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='beauty', y='score', hue='gender', data=data)
sns.lineplot(x=data_male['beauty'], y=model_male.predict(X_male), color='blue', label='Male Fit')
sns.lineplot(x=data_female['beauty'], y=model_female.predict(X_female), color='red', label='Female Fit')
plt.title('Score vs Beauty by Gender')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.legend()
plt.show()

# Output the R-squared values
r_squared_male = model_male.rsquared
r_squared_female = model_female.rsquared

print(f"R-squared for Male Model: {r_squared_male:.4f}")
print(f"R-squared for Female Model: {r_squared_female:.4f}")

# Equation of the line for male professors
slope_male = model_male.params[1]
intercept_male = model_male.params[0]
print(f"Equation for Male Professors: score = {intercept_male:.4f} + {slope_male:.4f} * beauty")
##################################################
#Question 26.1, Round 27 with threat_id: thread_I6SBHg1B14WEpm8vAv1szXDr
# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace 'your_data.csv' with your actual file path or data source
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
# Assuming 'avg_prof_evaluation' is the column for professor evaluation score,
# 'avg_beauty_rating' is for beauty rating, and 'gender' is coded as 0 and 1
X = data[['avg_beauty_rating', 'gender']]
X = sm.add_constant(X)  # Adding an intercept
y = data['avg_prof_evaluation']
score_bty_gender_fit = sm.OLS(y, X).fit()

# Output model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['avg_beauty_rating']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for average beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='avg_beauty_rating', y='avg_prof_evaluation', data=data, hue='gender', jitter=True, dodge=True)
plt.title('Scatterplot of Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 27 with threat_id: thread_NE5oRzb1oP5M3juBCoh0JslC
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is in a CSV file
# df = pd.read_csv('your_data.csv')

# Example data structure
# df should have columns: 'score', 'beauty', 'gender'
# df = pd.DataFrame({
#     'score': [some_values],
#     'beauty': [some_values],
#     'gender': ['male', 'female', ...]
# })

# Fit model for the entire dataset
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
y = df['score']

X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()

# Get R-squared
explained_variability = model.rsquared * 100  # Convert to percentage
print(f"Explained variability: {explained_variability:.2f}%")

# Equation of the line for male professors
model_coefficients = model.params
intercept = model_coefficients['const']
slope = model_coefficients['beauty']
equation_male = f"score = {intercept:.2f} + {slope:.2f} * beauty"
print(f"Equation for male professors: {equation_male}")

# Plotting the relationship
sns.lmplot(x='beauty', y='score', hue='gender', data=df, aspect=1.5)
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.show()

# Analyzing differences between genders
summary_by_gender = df.groupby('gender')['score'].describe()
print(summary_by_gender)
##################################################
#Question 26.1, Round 28 with threat_id: thread_rD9vORQ2JOFtpQgGQzNSCv5V
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Replace 'your_data.csv' with your actual data file
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variable
y = data['score']

# Add a constant (intercept) to the model
X = sm.add_constant(X)
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
# The intercept is the expected score when bty_avg=0 and gender=0 (baseline reference)
# The slope for bty_avg indicates the change in score for a unit increase in beauty rating
# The slope for gender indicates the difference in score between the reference group and the coded group

# Create scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 28 with threat_id: thread_FWav6qHPkQHOk5q1xPbHENB0
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv('path_to_your_data.csv')

# Fit model
model = sm.OLS(data['evaluation_score'], sm.add_constant(data[['beauty', 'gender']]))
results = model.fit()

# Print summary for model performance details
print(results.summary())

# Get R-squared (percent of variability explained)
r_squared = results.rsquared
print(f'R-squared (percent of variability explained): {r_squared * 100:.2f}%')

# Filter data for male professors
male_data = data[data['gender'] == 'male']

# Create model for male professors
male_model = sm.OLS(male_data['evaluation_score'], sm.add_constant(male_data['beauty']))
male_results = male_model.fit()

# Print the equation of the line for male professors
intercept, slope = male_results.params
print(f'Equation of the line for male professors: evaluation_score = {intercept:.2f} + {slope:.2f} * beauty')

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='beauty', y='evaluation_score', hue='gender', data=data)
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.title('Relationship Between Beauty and Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 29 with threat_id: thread_PpcB4QPt7jht0K1U5IB7r9zg
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data loading (replace 'your_data.csv' with your actual data file)
# df = pd.read_csv('your_data.csv')

# Assuming df is your DataFrame with columns 'score', 'bty_avg', and 'gender'

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']

# Adding a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Female' is the reference category

print(f'Intercept interpretation: The average professor evaluation score when beauty rating is 0 and gender is Female is {intercept}.')
print(f'Beauty rating slope interpretation: For each one-unit increase in beauty rating, the average professor evaluation score increases by {slope_bty}.')
print(f'Gender slope interpretation: Being Male compared to Female is associated with an increase of {slope_gender} in the average professor evaluation score.')

# Create a scatterplot with jitter
plt.figure(figsize=(10,6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, alpha=0.6, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 30 with threat_id: thread_LuDChkJErJ5iQ2QLGsAr6nRo
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with the path to your file)
# Make sure your data includes columns 'score', 'bty_avg', and 'gender'
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for gender
y = data['score']
X = sm.add_constant(X)  # Adds a constant term for the intercept

score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the results
intercept = score_bty_gender_fit.params[0]
slope_beauty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]  # Assuming it's for the 'gender_male'

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating (bty_avg): {slope_beauty}')
print(f'Slope for gender (if Gender is Male): {slope_gender}')  # Adjust according to your encoding

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 30 with threat_id: thread_HnXvqlGzI772AEVkCY10pqQw
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming you have a dataset 'data.csv' with columns 'beauty', 'evaluation_score', and 'gender'
data = pd.read_csv('data.csv')

# Fit the model for males
male_data = data[data['gender'] == 'male']
X_male = sm.add_constant(male_data['beauty'])  # Add a constant for intercept
y_male = male_data['evaluation_score']
model_male = sm.OLS(y_male, X_male).fit()

# Equation of the line for male professors
male_equation = f"Score = {model_male.params[0]:.4f} + {model_male.params[1]:.4f} * Beauty"
print(f"Equation for Male Professors: {male_equation}")

# Fit the model for females
female_data = data[data['gender'] == 'female']
X_female = sm.add_constant(female_data['beauty'])
y_female = female_data['evaluation_score']
model_female = sm.OLS(y_female, X_female).fit()

# Visualizing the relationship
plt.scatter(male_data['beauty'], male_data['evaluation_score'], color='blue', label='Male Professors')
plt.scatter(female_data['beauty'], female_data['evaluation_score'], color='red', label='Female Professors')

# Plotting the regression lines
plt.plot(male_data['beauty'], model_male.predict(X_male), color='blue', linewidth=2)
plt.plot(female_data['beauty'], model_female.predict(X_female), color='red', linewidth=2)

plt.title('Relationship between Beauty and Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()

# Print the variability explained
print(f"Male model explained variability: {model_male.rsquared:.2f}")
print(f"Female model explained variability: {model_female.rsquared:.2f}")
##################################################
#Question 26.0, Round 31 with threat_id: thread_oB4dHoNHmnGGQc84v9CqDHIh
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data creation
# Replace this with loading your actual data
data = {
    'bty_avg': np.random.rand(100) * 10,  # Average beauty rating
    'gender': np.random.choice(['Male', 'Female'], 100),  # Gender
    'score': np.random.rand(100) * 5  # Average professor evaluation score
}

df = pd.DataFrame(data)

# Convert gender to numeric ('Male': 1, 'Female': 0) for regression analysis
df['gender_numeric'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Define the independent variables (adding a constant for intercept)
X = df[['bty_avg', 'gender_numeric']]
X = sm.add_constant(X)

# Define the dependent variable
y = df['score']

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print model summary
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_numeric']

print(f'Intercept: {intercept}')
print(f'Slope (Beauty Rating): {slope_bty}')
print(f'Slope (Gender): {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Average Professor Evaluation Score by Average Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 31 with threat_id: thread_D63BxhhpuD5sUCtGZWVJzVvb
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (adjust the path as needed)
data = pd.read_csv('your_data_file.csv')  # replace with your dataset filename

# Sample structure of data must include 'score', 'beauty', 'gender' columns
# Assume 'score' is the evaluation score, 'beauty' is the beauty metric, and 'gender' has values 'male' or 'female'

# Fit the model for all professors
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
y = data['score']
model = sm.OLS(y, sm.add_constant(X)).fit()

# Print the summary to get explained variability
print(model.summary())

# Get the percentage of variability explained by the model (R-squared)
r_squared = model.rsquared * 100

# Fit the model for just male professors
male_data = data[data['gender'] == 'male']
X_male = male_data[['beauty']]
y_male = male_data['score']
male_model = sm.OLS(y_male, sm.add_constant(X_male)).fit()

# Get the equation of the line for male professors
male_coeff = male_model.params
male_equation = f"Score = {male_coeff[0]:.2f} + {male_coeff[1]:.2f} * Beauty"

# Visualize the relationship
plt.scatter(data[data['gender'] == 'male']['beauty'], data[data['gender'] == 'male']['score'], color='blue', label='Male Professors')
plt.scatter(data[data['gender'] == 'female']['beauty'], data[data['gender'] == 'female']['score'], color='red', label='Female Professors')
plt.plot(male_data['beauty'], male_model.predict(sm.add_constant(male_data[['beauty']])), color='blue', linewidth=2)
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.title('Beauty vs Evaluation Score')
plt.legend()
plt.show()

# Outcome summary
outcome = {
    "r_squared": r_squared,
    "male_equation": male_equation
}

print(outcome)
##################################################
#Question 26.0, Round 32 with threat_id: thread_cHhXbARYuJEPZzjLUtIGy8Vy
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
# Assuming your data is in a CSV file named 'data.csv'. Adjust the path as necessary.
# data = pd.read_csv('data.csv')

# For demonstration, let’s create a dummy dataset
# Replace this with your actual data loading process
data = pd.DataFrame({
    'bty_avg': np.random.uniform(1, 5, 100),    # Beauty ratings between 1 and 5
    'gender': np.random.choice(['Male', 'Female'], 100), # Randomly assigning gender
    'score': np.random.uniform(1, 5, 100)        # Evaluation scores between 1 and 5
})

# Convert gender to a numeric variable
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1}) # Male = 0, Female = 1

# Define the independent variables and dependent variable
X = data[['bty_avg', 'gender']]
y = data['score']

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', data=data, hue='gender', jitter=True, dodge=True, palette='Set1')
plt.title('Scatterplot of Average Professor Evaluation Score vs. Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 32 with threat_id: thread_yasaPYBZVELIXYLaPhsxIXIT
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data.csv') # Adjust the filename as necessary

# Assume df has columns 'beauty_score', 'evaluation_score', and 'gender'
# Fit a linear regression model for both genders
model = sm.OLS.from_formula('evaluation_score ~ beauty_score * gender', data=df).fit()

# Percent of variability explained by the model (R-squared)
r_squared = model.rsquared
percent_variability_explained = r_squared * 100

# Get the coefficients for male professors
male_coef = model.params['beauty_score'] + model.params['beauty_score:gender[T.Male]']  # Adjust this based on your dummy variable coding

# Prepare equation of line for male professors: y = mx + b
intercept = model.params['Intercept']
equation_male = f"y = {male_coef} * beauty_score + {intercept}"

# Create a plot to visualize the relationship
plt.figure(figsize=(12, 6))
sns.scatterplot(x='beauty_score', y='evaluation_score', hue='gender', data=df, alpha=0.6)
sns.lineplot(x=df['beauty_score'], y=model.predict(), color='blue', label='Regression Line')
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()

# Output results
print(f"Percent of variability explained by the model: {percent_variability_explained:.2f}%")
print(f"Equation of the line for male professors: {equation_male}")
##################################################
#Question 26.0, Round 33 with threat_id: thread_5GmjLmrYbEYSzP8tnpaXA3M2
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming 'data' is the DataFrame containing 'score_avg', 'bty_avg', and 'gender' columns
# Load your data into a pandas DataFrame
# data = pd.read_csv('your_data_file.csv')  # Uncomment and modify to load your data

# Fit multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Encode 'gender' as dummy variables
y = data['score_avg']
X = sm.add_constant(X)  # Add intercept term

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the regression results
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes:
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # If 'gender' is encoded as Male/Female

print(f'Intercept (expected score when beauty rating is 0 and gender is Female): {intercept}')
print(f'Slope of beauty rating (change in score for a one-unit increase in beauty rating): {slope_beauty}')
print(f'Slope of gender (change in score when going from Female to Male): {slope_gender}')

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True, alpha=0.6)
plt.title('Scatterplot of Average Professor Evaluation Score vs. Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 33 with threat_id: thread_xhu1UIjuZnImXb71Kk5MarSn
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('your_data.csv')  # replace 'your_data.csv' with your actual file path

# Example variables (replace these with your actual column names)
X = data[['beauty_score', 'gender']]  # Assuming 'gender' is a categorical variable
y = data['evaluation_score']

# Convert 'gender' to dummy variables
X = pd.get_dummies(X, drop_first=True)  # This will create a dummy variable for 'gender'

# Fit the model
model = sm.OLS(y, sm.add_constant(X)).fit()

# Print the summary
print(model.summary())

# Percentage of variability explained (R-squared)
percent_variability_explained = model.rsquared * 100
print(f"Percent of variability explained by model: {percent_variability_explained:.2f}%")

# Get coefficient for the male equation
male_intercept = model.params['const']  # Intercept
male_coef_beauty = model.params['beauty_score']
male_coef_gender = model.params['gender_M']  # Assuming 'M' is the male category
male_equation = f"y = {male_intercept:.2f} + {male_coef_beauty:.2f} * beauty_score + {male_coef_gender:.2f}"
print(f"Equation for male professors: {male_equation}")

# Plotting
for gender in ['M', 'F']:  # Assuming 'M' for Male, 'F' for Female
    subset = data[data['gender'] == gender]
    plt.scatter(subset['beauty_score'], subset['evaluation_score'], label=f'Gender: {gender}')

plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Beauty Score vs Evaluation Score by Gender')
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 34 with threat_id: thread_q9LNdMaeC30UNY7eKm1hg2eE
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Assuming you have a DataFrame named 'df' with the columns 'score_avg', 'bty_avg', and 'gender'
# df = pd.read_csv('your_file.csv')  # Load your data here

# Let's assume you have loaded your DataFrame into 'df'
# For this example, let's create a mock DataFrame
data = {
    'score_avg': [4.5, 3.8, 4.6, 4.4, 3.9, 4.7, 4.0, 3.5],
    'bty_avg': [4.0, 3.5, 4.5, 4.0, 3.0, 4.6, 4.1, 3.2],
    'gender': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female']
}
df = pd.DataFrame(data)

# Converting gender to a binary variable (0 for male, 1 for female for example)
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
y = df['score_avg']

# Add a constant for the intercept
X = sm.add_constant(X)
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope (Beauty Rating): {slope_bty}')
print(f'Slope (Gender): {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='bty_avg', y='score_avg', hue='gender', jitter=True, dodge=True, palette='Set1')
plt.title('Scatter Plot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.1, Round 35 with threat_id: thread_6eulMOCVaOjpZX8PrKJTo1AO
# Importing necessary libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment this line and replace with your actual dataset file

# Example dataset creation for demonstration purposes
# df = pd.DataFrame({
#     'score': [4.0, 3.5, 4.2, 3.8, 4.5, 3.9],
#     'bty_avg': [5, 4, 5, 3, 5, 4],
#     'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male']
# })

# Convert gender to categorical
df['gender'] = df['gender'].astype('category')

# Adding a constant for the intercept
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables into dummy/indicator variables
X = sm.add_constant(X)

y = df['score']

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Intercept and slopes interpretation
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Female' is the baseline

print(f"Intercept (constant): {intercept:.2f}")
print(f"Slope for beauty rating: {slope_bty:.2f} (for each unit increase in beauty rating, the score increases on average by {slope_bty:.2f})")
print(f"Slope for gender (Male): {slope_gender:.2f} (compared to Females, Males have an average score that is {slope_gender:.2f} higher)")

# Creating a scatterplot
plt.figure(figsize=(10, 6))
sns.violinplot(x='bty_avg', y='score', hue='gender', data=df, dodge=True, palette='muted', inner=None)
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='dark:.3', alpha=0.6)
plt.title('Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 35 with threat_id: thread_SwCYCM0E3OGSGz2oPvQXSepI
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('your_dataset.csv')  # Replace with the path to your dataset

# Fit the model for all professors and then separately for male professors
model_fit = sm.OLS(data['Score'], sm.add_constant(data['Beauty'])).fit()
male_data = data[data['Gender'] == 'Male']
male_model_fit = sm.OLS(male_data['Score'], sm.add_constant(male_data['Beauty'])).fit()

# Print the summary
print("Full model summary:")
print(model_fit.summary())
print("\nMale professors model summary:")
print(male_model_fit.summary())

# Plotting
sns.scatterplot(x='Beauty', y='Score', data=data, hue='Gender')
plt.plot(data['Beauty'], model_fit.predict(sm.add_constant(data['Beauty'])), label='All Professors', color='blue')
plt.plot(male_data['Beauty'], male_model_fit.predict(sm.add_constant(male_data['Beauty'])), label='Male Professors', color='red')
plt.legend()
plt.title('Score vs Beauty by Gender')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.show()
##################################################
#Question 26.1, Round 36 with threat_id: thread_GyLhxGFyMQZdcItLlBoLQAjw
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Ensure you have a DataFrame 'df' with columns 'score', 'bty_avg', and 'gender'

# Example: df = pd.read_csv("your_data.csv")

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the results
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Male' is one of the categories

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender (Male): {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, alpha=0.6)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 36 with threat_id: thread_JdZ2V1llTZEkkRZQTJFTKuRO
import pandas as pd
import statsmodels.api as sm

# Assuming 'data' is your DataFrame containing 'evaluation_score', 'beauty', and 'gender' columns.
# Load your dataset
# data = pd.read_csv('your_data.csv')

# Fit the model
model = sm.OLS.from_formula('evaluation_score ~ beauty * gender', data)
results = model.fit()

# Get the R-squared value which tells us the percentage of variability explained
r_squared = results.rsquared * 100  # converting to percentage

# Extract coefficients for male professors only
male_coefficients = results.params.filter(like='gender[T.Male]')
intercept = results.params['Intercept']
beauty_coef = results.params['beauty']
male_multiplier = male_coefficients.get('gender[T.Male]', 0)

# Equation for male professors
equation = f"Score = {intercept + male_multiplier} + {beauty_coef} * beauty"

# Summary of model results
print("Model Summary:\n", results.summary())
print(f"Percentage of variability explained: {r_squared}%")
print(f"Equation for male professors: {equation}")

# To compare effects, you may want to plot or further analyze
# Example of plotting omitted for brevity but can include regression lines based on fitted values
##################################################
#Question 26.1, Round 37 with threat_id: thread_Hiz90xE8P8hEv9jU8E2Zdw2x
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data creation (You may need to load your actual dataset)
# df = pd.read_csv('your_data.csv')  # Load your dataset
# Assuming the dataframe contains columns 'score', 'bty_avg', and 'gender'

# For demonstration purposes: creating a sample DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'score': np.random.uniform(0, 5, 100),
    'bty_avg': np.random.uniform(1, 10, 100),
    'gender': np.random.choice(['male', 'female'], 100)
})

# Convert 'gender' to a numerical format
df['gender_numeric'] = df['gender'].map({'male': 0, 'female': 1})

# Define the independent variables and dependent variable
X = df[['bty_avg', 'gender_numeric']]
y = df['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpretation
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_numeric']

print(f"Intercept: {intercept}")
print(f"Slope (Beauty Rating): {slope_bty}")
print(f"Slope (Gender - Female vs Male): {slope_gender}")

# Scatter plot with jitter
sns.scatterplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True)
plt.title('Scatter Plot of Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating (bty_avg)')
plt.ylabel('Average Professor Evaluation Score (score)')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 37 with threat_id: thread_2sdTf54MhvjwHGFy6ZO5jutl
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Sample Data Loading (please modify this to load your dataset)
# df = pd.read_csv('your_dataset.csv')

# Assume df has columns 'beauty_score', 'evaluation_score', and 'gender'
# Filter for male professors
male_professors = df[df['gender'] == 'male']
female_professors = df[df['gender'] == 'female']

# Fit the model for both genders
model = sm.OLS(df['evaluation_score'], sm.add_constant(df[['beauty_score', 'gender']])).fit()
print(f"Model Summary:\n{model.summary()}")

# Percent variability explained
r_squared = model.rsquared
print(f"Percent of variability explained: {r_squared * 100:.2f}%")

# Equation of the line for male professors
male_model = sm.OLS(male_professors['evaluation_score'], sm.add_constant(male_professors['beauty_score'])).fit()
alpha, beta = male_model.params
print(f"Equation for Male Professors: evaluation_score = {alpha:.2f} + {beta:.2f} * beauty_score")

# Visualizing the relationship
plt.scatter(male_professors['beauty_score'], male_professors['evaluation_score'], color='blue', label='Male Professors')
plt.scatter(female_professors['beauty_score'], female_professors['evaluation_score'], color='pink', label='Female Professors')

# Plotting regression lines
x_values = np.linspace(df['beauty_score'].min(), df['beauty_score'].max(), 100)
plt.plot(x_values, alpha + beta * x_values, color='blue')
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 38 with threat_id: thread_McSXbq6nbD1iAoEieF4O7aeH
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Assume df is your DataFrame containing the data
# Replace this with your actual data loading process
# e.g., df = pd.read_csv('your_file.csv')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']

# Adding a constant to the model (the intercept)
X = sm.add_constant(X)
model_score_bty_gender_fit = sm.OLS(y, X).fit()

# Display the model summary
print(model_score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = model_score_bty_gender_fit.params['const']
slope_bty = model_score_bty_gender_fit.params['bty_avg']
slope_gender = model_score_bty_gender_fit.params['gender_Male']  # Assuming 'Male' is one of the categories

print(f"Intercept: {intercept}")
print(f"Slope of beauty rating: {slope_bty}")
print(f"Slope of gender (Male): {slope_gender}")

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, palette="Set2", alpha=0.7)
plt.title("Average Professor Evaluation Score by Average Beauty Rating and Gender")
plt.xlabel("Average Beauty Rating")
plt.ylabel("Average Professor Evaluation Score")
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 38 with threat_id: thread_mTsBWHGiz1HLoY00Ubbq16cX
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Ensure that your dataset is correctly imported
data = pd.read_csv('your_data_file.csv')  # Replace 'your_data_file.csv' with the actual filename

# Example of common columns: 'beauty', 'score', 'gender'
data['gender'] = data['gender'].map({'male': 1, 'female': 0})  # Encoding gender if needed

# Define the model score_bty_gender_fit
X = data[['beauty', 'gender']]  # Independent variables
y = data['score']  # Dependent variable

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Summary of the model's results
print(model.summary())

# Calculate the percent of variability explained by the model
r_squared = model.rsquared
print(f'Percent of Variability Explained: {r_squared * 100:.2f}%')

# Get the equation of the line for just male professors
male_model = sm.OLS(y[data['gender'] == 1], X[data['gender'] == 1]).fit()
print(f'Equation for Male Professors: y = {male_model.params[0]:.2f} + {male_model.params[1]:.2f} * beauty + {male_model.params[2]:.2f}')

# Plotting the relationship
sns.lmplot(x='beauty', y='score', hue='gender', data=data, palette='Set1')
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 39 with threat_id: thread_EMJpWoWxPUgn4CH0lkiNwR3u
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# data = pd.read_csv('your_data_file.csv') # Uncomment and provide your data file path

# Assuming your dataset is loaded into a DataFrame named `data`
# Example DataFrame structure:
# data = pd.DataFrame({
#     'score_avg': [4.0, 3.8, 4.2, ...],
#     'bty_avg': [5.0, 6.0, 7.0, ...],
#     'gender': ['Male', 'Female', 'Female', ...]
# })

# Encoding gender as a binary variable
data['gender_encoded'] = data['gender'].map({'Male': 0, 'Female': 1})

# Defining the independent variables and the dependent variable
X = data[['bty_avg', 'gender_encoded']]
y = data['score_avg']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpreting coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 39 with threat_id: thread_kWC2dt9P55esJhghfRLrW8pw
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (this should be replaced with your actual data) 
# Assume 'data' has columns 'evaluation_score', 'beauty_rating', and 'gender'
data = pd.DataFrame({
    'evaluation_score': [3.5, 4.2, 2.8, 4.0, 5.0, 3.0, 3.8, 4.5, 2.5, 4.1],
    'beauty_rating': [5, 3, 4, 2, 5, 3, 4, 4, 2, 5],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
})

# OLS Regression Model
X = data[['beauty_rating']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(data['evaluation_score'], X).fit()

# Displaying the results
print(model.summary())

# Percent of variability explained
r_squared = model.rsquared * 100
print(f"Percent of variability explained by the model: {r_squared:.2f}%")

# Line equation for male professors
male_data = data[data['gender'] == 'Male']
model_male = sm.OLS(male_data['evaluation_score'], sm.add_constant(male_data['beauty_rating'])).fit()
intercept_male, slope_male = model_male.params
print(f"Equation of the line for male professors: y = {intercept_male:.2f} + {slope_male:.2f} * beauty_rating")

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='beauty_rating', y='evaluation_score', hue='gender', style='gender')
plt.plot(male_data['beauty_rating'], intercept_male + slope_male * male_data['beauty_rating'], color='blue', label='Male Fit Line')
plt.title('Beauty Rating vs Evaluation Score by Gender')
plt.xlabel('Beauty Rating')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 40 with threat_id: thread_2ZKd9bSf6VSVihWb7sPho0fQ
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset. Replace 'your_data.csv' with the correct data file.
# df = pd.read_csv('your_data.csv')

# Assuming your dataset has columns: 'score_avg', 'bty_avg', and 'gender'
# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X['gender'] = X['gender'].map({'male': 0, 'female': 1})  # Encode gender as 0 and 1
Y = df['score_avg']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the model
score_bty_gender_fit = sm.OLS(Y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Extracting coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Create scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, palette='Set1', dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 40 with threat_id: thread_Sg7q9XMfOgSCfyjmYiDuIktW
import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('your_data_file.csv')  # Adjust the file path as necessary

# Fit the model for all professors
X = data[['Beauty', 'Gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical 'Gender' into dummy variables
y = data['Score']
X = sm.add_constant(X)  # Add a constant for intercept

model = sm.OLS(y, X).fit()
print(model.summary())

# Percent of variability explained
rsquared = model.rsquared
print(f"The model explains {rsquared*100:.2f}% of the variability in score.")

# Equation for male professors
X_male = data[data['Gender'] == 'Male'][['Beauty']]
X_male = sm.add_constant(X_male)
model_male = sm.OLS(y[data['Gender'] == 'Male'], X_male).fit()
coef_male = model_male.params
print(f"Equation for Male Professors: Score = {coef_male[0]:.2f} + {coef_male[1]:.2f} * Beauty")

# Relationship between beauty and score by gender
sns.lmplot(data=data, x='Beauty', y='Score', hue='Gender', aspect=1.5)
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 41 with threat_id: thread_pO6zhi74NIuTuECdx7PkB5tS
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Assume data is in a DataFrame called df with columns: 'score', 'bty_avg', 'gender'
# For example, you can load your data from a CSV file or another source.
# df = pd.read_csv('your_data.csv')

# Create dummy variables for gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})  # Male as 0, Female as 1

# Fit multiple linear regression model
X = df[['bty_avg', 'gender']]
y = df['score']
X = sm.add_constant(X)  # Adds the intercept column

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret coefficients
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f'Intercept: {intercept}')
print(f'Slope (Beauty Rating): {slope_bty}')
print(f'Slope (Gender): {slope_gender}')

# Create scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bty_avg', y='score', hue='gender', jitter=True, alpha=0.7)
plt.title('Scatterplot of Average Professor Evaluation Scores by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.0, Round 41 with threat_id: thread_2mWTZNHHMwbnvQG5lkxel8mE
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('your_data.csv')  # Use your actual file

# Assuming 'score', 'beauty', and 'gender' are columns in your DataFrame
# Fit the model on the entire dataset
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
y = df['score']
model = sm.OLS(y, sm.add_constant(X)).fit()

# Get R-squared to find the percent of variability explained
r_squared = model.rsquared
percent_variability_explained = r_squared * 100

# Fit model for male professors only
male_df = df[df['gender'] == 'Male']
X_male = male_df[['beauty']]
y_male = male_df['score']
model_male = sm.OLS(y_male, sm.add_constant(X_male)).fit()

# Get the equation of the line (y = mx + b)
slope = model_male.params['beauty']
intercept = model_male.params['const']
equation = f"Score = {intercept:.2f} + {slope:.2f} * Beauty"

# Compare relationships between male and female professors
female_df = df[df['gender'] == 'Female']
model_female = sm.OLS(female_df['score'], sm.add_constant(female_df[['beauty']])).fit()

# Plotting the relationships
plt.scatter(male_df['beauty'], male_df['score'], color='blue', label='Male Professors')
plt.scatter(female_df['beauty'], female_df['score'], color='red', label='Female Professors')

# Prediction lines
plt.plot(male_df['beauty'], model_male.predict(sm.add_constant(X_male)), color='blue', linewidth=2, label='Male Fit Line')
plt.plot(female_df['beauty'], model_female.predict(sm.add_constant(female_df[['beauty']])), color='red', linewidth=2, label='Female Fit Line')

plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.legend()
plt.show()

# Output results
print(f"Percentage of variability in score explained by the model: {percent_variability_explained:.2f}%")
print(equation)
##################################################
#Question 26.1, Round 42 with threat_id: thread_DPzdtTwETvns9uJ0JJ9bqa5J
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
# Replace 'your_file.csv' with your actual file name
data = pd.read_csv('your_file.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical gender variable to dummy/indicator variables
y = data['score']

X = sm.add_constant(X)  # Adds a constant term to the predictor
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpretation of the model coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_M']  # Assuming 'M' represents males; adjust if necessary

print(f"Intercept: {intercept}")
print(f"Slope (Beauty Rating): {slope_bty}")
print(f"Slope (Gender): {slope_gender}")

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='bty_avg', y='score', hue='gender', jitter=True, alpha=0.6)
plt.title('Professor Evaluation Score by Average Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 42 with threat_id: thread_89J1rbKsUThf5lSyEpDOPcWG
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('path_to_your_data.csv')

# Example DataFrame structure
# df should have columns 'beauty', 'score', and 'gender'

# Fit the model
model = sm.OLS.from_formula('score ~ beauty * gender', data=df)
results = model.fit()

# Print the summary
print(results.summary())

# Get the R-squared value
r_squared = results.rsquared
print(f"R-squared: {r_squared:.2f}")

# Equation of the line for male professors
male_coeffs = results.params[['Intercept', 'beauty', 'gender[T.Male]', 'beauty:gender[T.Male]']]
male_equation = f"Score = {male_coeffs.values[0]:.2f} + {male_coeffs.values[1]:.2f}*Beauty + {male_coeffs.values[2]:.2f} + {male_coeffs.values[3]:.2f}*Beauty (Male)"
print(male_equation)

# Plotting
plt.figure(figsize=(10, 6))
for gender in df['gender'].unique():
    subset = df[df['gender'] == gender]
    plt.scatter(subset['beauty'], subset['score'], label=gender, alpha=0.5)

# Predicting Line
x = np.linspace(df['beauty'].min(), df['beauty'].max(), 100)
male_line = results.predict(exog=dict(beauty=x, gender='Male'))
female_line = results.predict(exog=dict(beauty=x, gender='Female'))

plt.plot(x, male_line, color='blue', label='Male Fit', linewidth=2)
plt.plot(x, female_line, color='red', label='Female Fit', linewidth=2)

plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Beauty vs. Evaluation Score by Gender')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 43 with threat_id: thread_rlAPasetmVS6o29EdcmE1VPz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load your data into a DataFrame
# Example: df = pd.read_csv('your_data.csv')
# For demonstration, replace this with your actual data
df = pd.DataFrame({
    'score_avg': np.random.rand(100) * 5,  # Simulated average professor evaluation scores
    'bty_avg': np.random.rand(100) * 10,    # Simulated average beauty ratings
    'gender': np.random.choice(['Male', 'Female'], 100)  # Simulated genders
})

# Convert gender to a category
df['gender'] = df['gender'].astype('category')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
y = df['score_avg']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()
print(model.summary())

# Intercept and slopes interpretation
intercept = model.params['const']
slope_bty = model.params['bty_avg']
slope_gender = model.params['gender_Male']  # Assumes 'Female' is the baseline

print(f"Intercept (average score when beauty rating is 0 and gender is Female): {intercept}")
print(f"Slope of beauty rating (increase in average score for each unit increase in beauty rating): {slope_bty}")
print(f"Slope of gender (difference in average score between Male and Female): {slope_gender}")

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, alpha=0.7, marker='o')
plt.title('Scatterplot of Average Scores by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 43 with threat_id: thread_ZlyQso9MArRzUvK5pNsYG1N8
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Example data (replace this with your actual data)
df = pd.DataFrame({
    'beauty': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
    'score': [2, 3, 4, 5, 6, 6, 5, 4, 3, 2],
    'gender': ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'female']
})

# Fit the model
X = df[['beauty']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['score']
model = sm.OLS(y, X).fit()

# Variability explained (R-squared)
r_squared = model.rsquared
print(f'Percentage of Variability Explained: {r_squared * 100:.2f}%')

# Get the coefficients for male professors
male_df = df[df['gender'] == 'male']
X_male = male_df[['beauty']]
X_male = sm.add_constant(X_male)
y_male = male_df['score']
male_model = sm.OLS(y_male, X_male).fit()

# Equation of the line: y = mx + b
intercept = male_model.params[0]
slope = male_model.params[1]
print(f'Equation for male professors: score = {intercept:.2f} + {slope:.2f} * beauty')

# Create visualizations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='beauty', y='score', hue='gender', style='gender', markers=['o', 's'])
plt.plot(male_df['beauty'], male_model.predict(X_male), color='blue', label='Male Fit', linewidth=2)
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 44 with threat_id: thread_NaE4n0aphp1hegn2CUryorL4
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your data
# Assuming the DataFrame is named 'data' and has columns 'score', 'bty_avg', and 'gender'
data = pd.read_csv('your_data_file.csv')  # Replace with your actual data file

# Step 2: Prepare your data
# Convert 'gender' to a categorical variable if it's not already
data['gender'] = data['gender'].astype('category')

# Step 3: Define predictor variables and response variable
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical gender into dummy variables
y = data['score']

# Step 4: Add a constant to the predictor variables
X = sm.add_constant(X)

# Step 5: Fit the regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Output the summary of the model
print(score_bty_gender_fit.summary())

# Intercept and slopes interpretation
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2:]  # If 'gender' has more than one category

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
for g, slope in zip(score_bty_gender_fit.params.index[2:], slope_gender):
    print(f"Slope for gender {g}: {slope}")

# Step 6: Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 44 with threat_id: thread_9YVjYcdf2RKcUenQff72YsbW
import statsmodels.api as sm
import pandas as pd

# Assuming 'df' is your DataFrame containing 'beauty' and 'evaluation_score' with a 'gender' column
# Sample data loading (replace with your actual data loading mechanism)
# df = pd.read_csv('your_data.csv')

# Fit the model
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Encode categorical variables
y = df['evaluation_score']
model = sm.OLS(y, sm.add_constant(X)).fit()

# Summary of the model
print(model.summary())

# Extract percentage of variability explained by the model
r_squared = model.rsquared
print(f"R²: {r_squared * 100:.2f}%")

# Find coefficients for male professors
if 'gender_male' in X.columns:
    male_coef = model.params['beauty'] + model.params['gender_male'] * X['gender_male'].mean()
else:
    male_coef = model.params['beauty']

# Equation of the line for male professors
print(f"Equation for male professors: y = {male_coef} * beauty + {model.params['const']}")

# Compare coefficients for beauty between genders
# This assumes 'gender_female' is another dummy variable
female_coef = model.params['beauty']
print(f"Coefficient for beauty (Female): {female_coef}")
print(f"Coefficient for beauty (Male): {male_coef}")
##################################################
#Question 26.1, Round 45 with threat_id: thread_vG5KGPmSQKMLREGRNj7DWMYT
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with the actual dataset file)
data = pd.read_csv('your_data.csv')

# Assume the columns are named 'score_avg' for average professor evaluation score, 
# 'bty_avg' for average beauty rating, and 'gender' for gender
# Convert gender to numerical values if necessary (e.g., Male=0, Female=1)
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# Define the independent variables (X) and the dependent variable (y)
X = data[['bty_avg', 'gender']]
y = data['score_avg']

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='best', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.0, Round 45 with threat_id: thread_5FLfluxDOi3tbYM0p7qnKTme
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data.csv')  # replace with your data file

# Model fitting
# Assuming 'score' is your dependent variable and 'beauty' and 'gender' are your independent variables
df['gender'] = df['gender'].map({'male': 0, 'female': 1})  # Encoding gender as binary
X = df[['beauty', 'gender', 'beauty:gender']]  # Interaction term can be created if needed
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['score']

model = sm.OLS(y, X).fit()

# Print model summary to see R-squared value
print(model.summary())

# Calculate variance explained
r_squared = model.rsquared * 100  # Percentage of variance explained

# Equation of the line for male professors
male_equation = f"Score = {model.params[0]} + {model.params[1]}*Beauty + {model.params[2]}*0 + {model.params[3]}*0"
print("Equation for Male Professors:", male_equation)

# Visualizing relationship
sns.lmplot(x='beauty', y='score', data=df, hue='gender', markers=["o", "x"])
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.show()

# Relationship comparison
# You can look at interaction effects 
interaction_effects = model.summary().tables[1]  # Coefficients table
print(interaction_effects)
##################################################
#Question 26.1, Round 46 with threat_id: thread_cJba8N9UztXTCCLxuR3ZCxh1
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# Assuming `df` contains the columns 'score', 'bty_avg', and 'gender'
# Create dummy variables for gender
df['gender'] = df['gender'].map({'male': 0, 'female': 1})  # Encoding gender as binary (0 for male, 1 for female)

# Prepare the data
X = df[['bty_avg', 'gender']]
y = df['score']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and the slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f"Intercept Interpretation: The average professor evaluation score when beauty rating is 0 and gender is male (0) is approximately {intercept}.")
print(f"Slope for Beauty Rating Interpretation: For each one-unit increase in the average beauty rating, the average professor evaluation score increases by approximately {slope_bty}.")
print(f"Slope for Gender Interpretation: Changing gender from male (0) to female (1) is associated with an increase of approximately {slope_gender} in the average professor evaluation score.")

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='bty_avg', y='score', hue='gender', jitter=True, palette="Set1", alpha=0.6)
plt.title('Average Professor Evaluation Score by Average Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper left')
plt.show()
##################################################
#Question 26.0, Round 47 with threat_id: thread_DEnepm9vNdmyWfQXkIaV1v8V
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data loading
# Replace 'data.csv' with your own data file
# df = pd.read_csv('data.csv')

# Assuming df is your DataFrame and has the required columns
# Example DataFrame structure:
# df = pd.DataFrame({
#     'score_avg': [...],      # average professor evaluation score
#     'bty_avg': [...],        # average beauty rating
#     'gender': [...]          # gender indicators (e.g., 'male', 'female')
# })

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X['gender'] = pd.get_dummies(X['gender'], drop_first=True)  # Convert categorical variable to dummy variable
X = sm.add_constant(X)  # Adds an intercept term
y = df['score_avg']

model = sm.OLS(y, X).fit()
score_bty_gender_fit = model

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = model.params['const']
slope_bty = model.params['bty_avg']
slope_gender = model.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope (Beauty Rating): {slope_bty}')
print(f'Slope (Gender): {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, alpha=0.7)
plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 47 with threat_id: thread_F9jvp8wBe2S3DfIWtnHS2Jjb
import pandas as pd
import statsmodels.api as sm

# Sample data creation (replace this with your actual data)
data = {
    'gender': ['male', 'female', 'male', 'female', 'male'],
    'beauty': [3, 4, 2, 5, 3],
    'score': [4, 5, 3, 4, 5]
}
df = pd.DataFrame(data)

# Filter for male professors
male_professors = df[df['gender'] == 'male']

# Define the independent variable (beauty) and dependent variable (score)
X = male_professors['beauty']
y = male_professors['score']

# Add a constant to the independent variable for the intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())
##################################################
#Question 26.0, Round 48 with threat_id: thread_dTHKX4Z7A89NOPPb0OtsSMS2
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data loading (replace 'your_data.csv' with your actual data file)
# data = pd.read_csv('your_data.csv')

# For the purpose of this example, let's create some sample data
data = pd.DataFrame({
    'score': [3.5, 4.0, 3.0, 4.5, 4.0, 3.5, 3.0, 4.5, 4.0, 2.5],
    'bty_avg': [5, 6, 4, 7, 6, 5, 3, 8, 7, 2],
    'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
})

# Encoding categorical variable (gender)
data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})

# Defining independent variables and adding a constant for intercept
X = data[['bty_avg', 'gender']]
X = sm.add_constant(X)

# Dependent variable
y = data['score']

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Summarizing the model
print(score_bty_gender_fit.summary())

# Interpreting the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Creating the scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='bty_avg', y='score', hue='gender', jitter=True, palette='deep')
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.grid()
plt.show()
##################################################
#Question 26.1, Round 48 with threat_id: thread_XI8ZhsnADNRRMd4LJaaM8Fvt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('your_data.csv')  # Uncomment this line and provide your dataset

# Example dataset creation (remove this line and use your actual data)
data = {
    'beauty': np.random.rand(100) * 10,
    'score': np.random.rand(100) * 10,
    'gender': np.random.choice(['Male', 'Female'], 100)
}
df = pd.DataFrame(data)

# Fit the regression model for beauty and gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})  # Encode gender
X = df[['beauty', 'gender', 'beauty:gender']]
X['beauty:gender'] = df['beauty'] * df['gender']
y = df['score']

# Add a constant for intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Show the summary of the model
print(model.summary())

# Calculate R-squared to see the percentage of variability explained
r_squared = model.rsquared
print(f"Percentage of Variability Explained: {r_squared * 100:.2f}%")

# Equation of the line for male professors
male_coeffs = model.params[:2]  # Constant and beauty coefficient
male_equation = f"Score = {male_coeffs[0]:.2f} + {male_coeffs[1]:.2f} * Beauty"
print(f"Equation for Male Professors: {male_equation}")

# Plot the relationship
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='beauty', y='score', hue='gender')
sns.regplot(data=df[df['gender'] == 1], x='beauty', y='score', scatter=False, color='blue', label='Male')
sns.regplot(data=df[df['gender'] == 0], x='beauty', y='score', scatter=False, color='orange', label='Female')
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.0, Round 49 with threat_id: thread_s8HNedHMSlNGSDXmamU7HsXV
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Adjust the filepath for your actual data
data = pd.read_csv('your_data_file.csv') # replace with your data source

# Ensure gender is a categorical variable
data['gender'] = data['gender'].astype('category')

# Define the independent variables (X) and dependent variable (y)
X = data[['bty_avg', 'gender']]
y = data['score']

# One-hot encoding for categorical variables (gender)
X = pd.get_dummies(X, drop_first=True)

# Fit the model
score_bty_gender_fit = sm.OLS(y, sm.add_constant(X)).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret intercept and slopes
intercept = score_bty_gender_fit.params[0]
beauty_slope = score_bty_gender_fit.params['bty_avg']
gender_slope = score_bty_gender_fit.params['gender_Male']  # Adjust based on your encoding

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {beauty_slope}')
print(f'Slope for gender (Male): {gender_slope}')

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='bty_avg', y='score', hue='gender', jitter=True, alpha=0.7)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.grid(True)
plt.show()
##################################################
#Question 26.1, Round 49 with threat_id: thread_mviXnGGlnNebAkebvTGqiuxJ
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data creation - replace this with your actual dataset
# df = pd.read_csv('your_data.csv') # Load your dataset
# Example of data structure you may have
data = {
    'beauty': [5, 7, 6, 8, 9, 4, 6, 7, 8, 5],
    'score': [2, 7, 6, 8, 9, 3, 5, 7, 8, 2],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
}
df = pd.DataFrame(data)

# Fit the model
model = sm.OLS(df['score'], sm.add_constant(df[['beauty', 'gender']]))
results = model.fit()

# Summary of the model, including R-squared value
print(results.summary())

# Extract R-squared value to determine percent variability explained
r_squared = results.rsquared
print(f'R-squared: {r_squared * 100:.2f}% variability explained by the model.')

# Prepare the equations for males and females
male_params = results.params[results.params.index.str.contains('Male')]
female_params = results.params[results.params.index.str.contains('Female')]

male_eq = f"y = {male_params[0]:.2f} + {male_params[1]:.2f}*X_beauty" # Intercept + slope
female_eq = f"y = {female_params[0]:.2f} + {female_params[1]:.2f}*X_beauty" # Intercept + slope
print(f'Equation for Male Professors: {male_eq}')
print(f'Equation for Female Professors: {female_eq}')

# Visualizing the data
sns.lmplot(x='beauty', y='score', hue='gender', data=df)
plt.title('Beauty vs Evaluation Score by Gender')
plt.show()
##################################################
#Question 26.0, Round 50 with threat_id: thread_NKv6WLLf7DW07np1MMF8zHO5
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# df = pd.read_csv('your_data_file.csv')  # Uncomment and modify this line to load your data

# Ensure 'gender' is encoded as a categorical variable
df['gender'] = df['gender'].astype('category')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy/indicator variables
y = df['score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

score_bty_gender_fit = sm.OLS(y, X).fit()  # Fit the model

# Print the summary to interpret coefficients
print(score_bty_gender_fit.summary())

# Intercept interpretation
intercept = score_bty_gender_fit.params['const']
print(f'Intercept: {intercept}')

# Slope interpretations
slope_bty = score_bty_gender_fit.params['bty_avg']
print(f'Slope for beauty rating: {slope_bty}')

slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Female' is the reference category
print(f'Slope for male category (compared to female): {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Scatter plot of Score by Beauty Rating, Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 50 with threat_id: thread_V0NegR9oE7H5FCIcSTE0mttL
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Assuming df is your DataFrame containing 'evaluation_score', 'beauty_score', and 'gender'
# df should have columns: 'evaluation_score', 'beauty', 'gender'

# Fit the model
X = df[['beauty_score', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender into dummy variables
y = df['evaluation_score']
model = sm.OLS(y, sm.add_constant(X)).fit()

# Get R-squared
r_squared = model.rsquared

# Summary of the model to get the coefficients
summary = model.summary()

# Extracting the equation for male professors if the male gender is represented by 'gender_male'
male_coefficients = model.params[['const', 'beauty_score', 'gender_male']]  # Adjust as necessary
male_equation = f"Score = {male_coefficients['const']} + {male_coefficients['beauty_score']} * Beauty + {male_coefficients['gender_male']}"

print(f"R-squared (percent of variability explained): {r_squared * 100:.2f}%")
print(f"Equation for male professors: {male_equation}")

# Analyzing the relationship between genders
female_coefficients = model.params[['const', 'beauty_score', 'gender_female']]  # Adjust as necessary
print(f"Difference in coefficients - Male: {male_coefficients['beauty_score']}, Female: {female_coefficients['beauty_score']}")
##################################################
#Question 26.0, Round 51 with threat_id: thread_uS3EkzO69BrIWmufCy3rfB0W
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Load your data into a DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment and replace with your actual file path

# Fit a multiple linear regression model
score_bty_gender_fit = ols('avg_professor_evaluation_score ~ avg_beauty_rating + C(gender)', data=df).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['Intercept']
beauty_slope = score_bty_gender_fit.params['avg_beauty_rating']
gender_slope = score_bty_gender_fit.params['C(gender)[T.Male]']  # Assuming 'Male' and 'Female' as the two categories

print(f'Intercept: {intercept}')
print(f'Coefficient for average beauty rating: {beauty_slope}')
print(f'Coefficient for gender (Male): {gender_slope}')

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='avg_beauty_rating', y='avg_professor_evaluation_score', hue='gender', data=df, jitter=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 51 with threat_id: thread_qUXKRZaX5HhAUqgjsfp3ztZe
import pandas as pd
import statsmodels.api as sm

# Sample data loading, you need to replace it with your actual dataset
# data = pd.read_csv('your_data.csv')

# Example DataFrame structure
data = pd.DataFrame({
    'score': [/* your scores here */],
    'beauty': [/* beauty ratings here */],
    'gender': [/* 'male' or 'female' */]
})

# Fit the model
model = sm.OLS(data['score'], sm.add_constant(data[['beauty', 'gender']])).fit()

# Percentage of variability explained
r_squared = model.rsquared
percent_explained = r_squared * 100

# Male only model
male_data = data[data['gender'] == 'male']
male_model = sm.OLS(male_data['score'], sm.add_constant(male_data['beauty'])).fit()
male_line_eq = f"Score = {male_model.params[0]:.2f} + {male_model.params[1]:.2f} * Beauty"

# Outputs
print(f"Percent variability explained: {percent_explained:.2f}%")
print(f"Equation of the line for male professors: {male_line_eq}")

# Compare relationships
female_data = data[data['gender'] == 'female']
female_model = sm.OLS(female_data['score'], sm.add_constant(female_data['beauty'])).fit()

print(f"Score for females: {female_model.params[0]:.2f} + {female_model.params[1]:.2f} * Beauty")
##################################################
#Question 26.0, Round 52 with threat_id: thread_sDGKABRzu7ryCVsPSYv1YSvr

Now, here’s the Python code:

##################################################
#Question 26.1, Round 52 with threat_id: thread_M2E1c2lRKAZDDaULNFOzOIvx
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data.csv')  # Example for loading data

# Assuming the dataset is available as DataFrame `df` with columns 'beauty', 'evaluation_score', and 'gender'
# Fit the model
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy variables
y = df['evaluation_score']
model = sm.OLS(y, X).fit()

# Display the model summary
print(model.summary())

# Percentage of variance explained (R-squared)
explained_variance = model.rsquared * 100
print(f"Percentage of variance explained by the model: {explained_variance:.2f}%")

# Equation for males only (assuming 'gender_male' is the dummy variable)
coef = model.params
male_equation = f"evaluation_score = {coef['const']:.2f} + {coef['beauty']:.2f} * beauty"
print(f"Equation for male professors: {male_equation}")

# Visualization of the relationship for male and female
plt.figure(figsize=(12, 6))
sns.scatterplot(x='beauty', y='evaluation_score', hue='gender', data=df)
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend(title='Gender')
plt.show()

# Interpretation of the results
if coef['gender_male'] > coef['gender_female']:
    relationship = "Males have a stronger positive relationship between beauty and evaluation score."
else:
    relationship = "Females have a stronger positive relationship between beauty and evaluation score."
print(relationship)
##################################################
#Question 26.0, Round 53 with threat_id: thread_WVYGkgbizlfKlMwKXuBTn4eu
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Assuming your data is in a pandas DataFrame named 'df'
# df = pd.read_csv('your_data.csv')  # Uncomment and update the path to your dataset

# Example DataFrame setup (remove this and uncomment the loading line to use your dataset directly)
# df = pd.DataFrame({
#     'score': [4.2, 3.8, 4.5, 4.0, 3.9],
#     'bty_avg': [3.5, 3.2, 4.0, 3.8, 3.7],
#     'gender': ['M', 'F', 'M', 'F', 'M']
# })

# Encoding gender as a binary variable (0 for Female, 1 for Male)
df['gender_encoded'] = np.where(df['gender'] == 'M', 1, 0)

# Fit multiple linear regression model
X = df[['bty_avg', 'gender_encoded']]
y = df['score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret intercept and slopes
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f'Intercept (Constant): {intercept}')
print(f'Slope (Beauty Rating): {slope_bty}')
print(f'Slope (Gender): {slope_gender}')

# Create scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 54 with threat_id: thread_1JEagXIo2u1QA6SJu3wG0sna
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming data is in a DataFrame called df with columns 'score', 'bty_avg', and 'gender'
df = pd.read_csv('your_data_file.csv')  # Load your dataset here

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']

# Adding a constant for the intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Summary of the regression results
print(model.summary())

# Interpret the intercept and slopes
intercept = model.params['const']
slope_beauty = model.params['bty_avg']
slope_gender = model.params['gender_Male']  # assuming 'gender' has values 'Male' and 'Female'

print(f'Intercept: {intercept}')
print(f'Slope for average beauty rating: {slope_beauty}')
print(f'Slope for gender (Male): {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bty_avg', y='score', hue='gender', jitter=True, alpha=0.7)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 54 with threat_id: thread_smxF7t72QxXgxsSpAPP9RbgP
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (update the path as necessary)
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with the actual file name

# Assuming 'beauty', 'evaluation_score', and 'gender' are columns in your dataset
df['is_male'] = np.where(df['gender'] == 'male', 1, 0)

# Fit the model
X = df[['beauty', 'is_male']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['evaluation_score']
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

# Calculate R-squared (percent variability explained)
r_squared = model.rsquared * 100
print(f"Percent of the variability in score explained by the model: {r_squared:.2f}%")

# Get the equation for male professors
male_coef = model.params['is_male']
beauty_coef = model.params['beauty']
intercept = model.params['const']
print(f"Equation for male professors: score = {intercept:.2f} + {beauty_coef:.2f} * beauty + {male_coef:.2f} * is_male")

# Plotting
sns.lmplot(x='beauty', y='evaluation_score', hue='gender', data=df, aspect=1.5)
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 55 with threat_id: thread_nG539ulfs5CjlAFiRnsgd1EL
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame 'df' with the columns 'bty_avg', 'gender', and 'score'
# Example: df = pd.read_csv('your_data_file.csv')

# Prepare the data
df['gender'] = df['gender'].astype('category')  # Ensure 'gender' is a categorical variable
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
y = df['score']

# Fit the model
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
score_bty_gender_fit = model

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = model.params['const']
slope_bty = model.params['bty_avg']  # Assuming bty_avg is in the dummy DataFrame
slope_gender = model.params['gender_second_value']  # Use the name corresponding to the second category

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, dodge=True, jitter=True, alpha=0.7)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 55 with threat_id: thread_Zf0utkZrjbZfLme95wZjva0L
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame 'df' with columns 'beauty_score', 'eval_score', and 'gender'
# Replace these with actual data
# df = pd.read_csv("your_data.csv")

# Fit the model
X = df[['beauty_score', 'gender']]  # Ensure 'gender' is one-hot encoded or categorical
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
y = df['eval_score']

model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Calculate explained variance
explained_variance = model.rsquared * 100  # Convert to percentage
print(f"Percent of the variability explained by the model: {explained_variance:.2f}%")

# Get the coefficients for the equation of the line for male professors
coefficients = model.params
intercept = coefficients[0]
slope = coefficients['beauty_score']  # Adjust based on how your columns are named
equation_male = f"eval_score = {intercept:.2f} + {slope:.2f} * beauty_score"
print(f"Equation of the line for male professors: {equation_male}")

# Plotting the relationship between beauty and evaluation score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='beauty_score', y='eval_score', hue='gender', alpha=0.7)
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.axhline(y=model.predict([1, 0]), color='blue', linestyle='--')  # Female
plt.axhline(y=model.predict([0, 1]), color='orange', linestyle='--')  # Male
plt.show()
##################################################
#Question 26.1, Round 56 with threat_id: thread_KMNvonTGSCFKi5703sTNaIkA
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into a pandas DataFrame
# Assuming your DataFrame is named df and contains columns: 'score', 'bty_avg', and 'gender'
# df = pd.read_csv('your_data_file.csv')  # Uncomment and modify this line to load your data

# Defining predictor variables and response variable
X = df[['bty_avg', 'gender']]
X['gender'] = X['gender'].map({'Male': 0, 'Female': 1})  # Encoding gender as binary
y = df['score']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpreting the model coefficients
intercept = score_bty_gender_fit.params['const']
bty_slope = score_bty_gender_fit.params['bty_avg']
gender_slope = score_bty_gender_fit.params['gender']
print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {bty_slope}')
print(f'Slope for Gender: {gender_slope}')

# Creating a scatter plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette={'Male': 'blue', 'Female': 'pink'})
plt.title('Scatterplot of Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 56 with threat_id: thread_rhoYMHhxkb5rHSW4s48L3vaj
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (Make sure to update the path to your data file)
data = pd.read_csv('your_data_file.csv')

# Fit the model for beauty vs evaluation score including gender as a factor
model = sm.OLS(data['evaluation_score'], sm.add_constant(data[['beauty_score', 'gender']])).fit()

# Percentage of variability explained
r_squared = model.rsquared * 100
print(f"Percentage of Variability Explained: {r_squared:.2f}%")

# Equation of the line for male professors
male_data = data[data['gender'] == 'male']  # Assuming gender is like 'male' or 'female'
X_male = sm.add_constant(male_data['beauty_score'])
model_male = sm.OLS(male_data['evaluation_score'], X_male).fit()

intercept = model_male.params[0]
slope = model_male.params[1]
print(f"Equation of the line for male professors: y = {intercept:.2f} + {slope:.2f} * x")

# Relationship visualization
sns.lmplot(data=data, x='beauty_score', y='evaluation_score', hue='gender', markers=["o", "X"])
plt.title('Relationship between Beauty and Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 57 with threat_id: thread_u26gTb7TlmCXGoIKxThq3bdS
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your data
# Assuming your DataFrame is named 'df' and contains columns 'score', 'bty_avg', and 'gender'
# df = pd.read_csv('path_to_your_data.csv') # Uncomment this line to load your data

# Create dummy variables for gender
df['gender'] = df['gender'].map({'male': 0, 'female': 1}) # Mapping male to 0 and female to 1

# Prepare the independent variables and dependent variable
X = df[['bty_avg', 'gender']]
y = df['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, dodge=True, jitter=True, palette='Set1', alpha=0.7)
plt.title('Scatterplot of Score by Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='best', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.0, Round 57 with threat_id: thread_I8JdvpO4awqBXC366kYUiPwg
import pandas as pd
import statsmodels.api as sm

# Assuming 'data' is a DataFrame containing your data
data = pd.read_csv('your_dataset.csv')  # Adjust the filename as necessary

# Fit the model
model = sm.OLS(data['score'], sm.add_constant(data[['beauty', 'gender']])).fit()

# Get summary of the model
model_summary = model.summary()
print(model_summary)

# Calculate the percentage of variability explained
r_squared = model.rsquared
percent_variability_explained = r_squared * 100

# Separate analysis for male professors
male_data = data[data['gender'] == 'male']
male_model = sm.OLS(male_data['score'], sm.add_constant(male_data[['beauty']])).fit()
male_equation = male_model.params

print(f"Percentage of variability explained: {percent_variability_explained:.2f}%")
print(f"Equation for male professors: score = {male_equation[0]:.4f} + {male_equation[1]:.4f} * beauty")

# Optional: Plot to visualize the relationship
import matplotlib.pyplot as plt
plt.scatter(data['beauty'], data['score'], alpha=0.5, c=data['gender'].map({'male': 'blue', 'female': 'red'}))
plt.xlabel('Beauty Rating')
plt.ylabel('Evaluation Score')
plt.title('Beauty vs Evaluation Score by Gender')
plt.show()
##################################################
#Question 26.1, Round 58 with threat_id: thread_ohbfrp9zijh3rHu0Emt9lKSW
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data, assuming it's in a CSV format
# Replace 'your_data.csv' with the path to your file
data = pd.read_csv('your_data.csv')

# Encoding gender as numeric values if necessary
data['gender_encoded'] = data['gender'].map({'male': 0, 'female': 1})

# Define the independent variables and add a constant for the intercept
X = data[['bty_avg', 'gender_encoded']]
X = sm.add_constant(X)

# Define the dependent variable
y = data['score']

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f"Intercept: {intercept:.4f}")
print(f"Slope for Beauty Rating (bty_avg): {slope_bty:.4f}")
print(f"Slope for Gender: {slope_gender:.4f}")

# Scatterplot with jitter according to gender
plt.figure(figsize=(8, 6))
sns.stripplot(x='bty_avg', y='score', data=data, hue='gender', jitter=True, dodge=True, palette='Set1')
plt.title('Scatterplot of Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 58 with threat_id: thread_bydfmfDpkbWoUySXDffNnzyA
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Assuming your dataset is a pandas DataFrame named 'df' with columns 'beauty', 'gender', and 'score'
df = pd.read_csv('your_dataset.csv')

# Convert gender into binary format
df['gender'] = np.where(df['gender'] == 'male', 1, 0)

# Create interaction term
df['bty_gender'] = df['beauty'] * df['gender']

# Define independent variables and dependent variable
X = df[['beauty', 'gender', 'bty_gender']]
y = df['score']

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# To get the summary statistics
print(model.summary())

# Extracting R-squared and equation for male professors
r_squared = model.rsquared
coef = model.params
equation_male = f"Score = {coef[0]} + {coef[1]}(beauty) + {coef[2]}(gender) + {coef[3]}(beauty * gender)"

print(f"Percentage of Variability Explained: {r_squared * 100:.2f}%")
print(f"Equation of the line for male professors: {equation_male}")
##################################################
#Question 26.1, Round 59 with threat_id: thread_jzjAR1YyW4vLIP2oLGNgyOmU
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

# Load your dataset - replace 'your_dataset.csv' with the actual file path
df = pd.read_csv('your_dataset.csv')

# Assuming the dataset has columns 'score', 'bty_avg', and 'gender'
# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variables
y = df['score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'gender_Male' is created for male gender

print(f"Intercept (constant term): {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender (Male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', data=df, hue='gender', jitter=True, dodge=True, alpha=0.7)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 59 with threat_id: thread_HZbep2TQqA9uAbcZED7uJVhy
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Sample data, replace this with your actual data
data = {
    'gender': ['male', 'female', 'male', 'female'],
    'beauty_score': [7, 8, 5, 6],
    'evaluation_score': [3, 5, 2, 6]
}
df = pd.DataFrame(data)

# Fit the linear regression model
X = df[['beauty_score', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
y = df['evaluation_score']

model = sm.OLS(y, sm.add_constant(X)).fit()

# Summary of the model
print(model.summary())

# Percentage of variability explained
r_squared = model.rsquared * 100
print(f"Percentage of variability explained by the model: {r_squared:.2f}%")

# Equation for male professors only
male_profs = df[df['gender'] == 'male']
X_male = male_profs[['beauty_score']]
X_male = sm.add_constant(X_male)
model_male = sm.OLS(male_profs['evaluation_score'], X_male).fit()

# Display the line equation
coeffs = model_male.params
print(f"Equation for male professors: y = {coeffs[0]:.2f} + {coeffs[1]:.2f} * beauty_score")

# Compare beauty and evaluation between genders
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='gender', y='evaluation_score', data=df)
plt.title('Evaluation Scores by Gender')
plt.show()

# Calculate and display beauty and evaluation score relationships
mean_scores = df.groupby('gender')[['beauty_score', 'evaluation_score']].mean()
print(mean_scores)
##################################################
#Question 26.1, Round 60 with threat_id: thread_v9JbsnRT390gYYkUWofOGvk3
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
# Make sure to replace 'your_data.csv' with the actual file path
data = pd.read_csv('your_data.csv')

# Prepare the data
# Assuming your dataset has columns named 'score', 'bty_avg', and 'gender'
data['gender'] = data['gender'].map({'male': 1, 'female': 0})  # Encoding gender if necessary

# Define the independent variables and the dependent variable
X = data[['bty_avg', 'gender']]
y = data['score']

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty_avg = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty_avg}')
print(f'Slope for gender: {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(data=data, x='bty_avg', y='score', hue='gender', jitter=True, dodge=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='best')
plt.show()
##################################################
#Question 26.0, Round 60 with threat_id: thread_s71CKQSdQZ5k8yrJefeBlHhQ
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample Data Loading (replace 'your_data.csv' with your actual data source)
data = pd.read_csv('your_data.csv')  # Make sure your data has columns 'beauty', 'score' and 'gender'

# Fit the full model
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy/indicator variables
y = data['score']

model = sm.OLS(y, sm.add_constant(X)).fit()

# R-squared (percentage of variability explained by the model)
explained_variability = model.rsquared * 100
print(f"Percent of the variability in score explained by the model: {explained_variability:.2f}%")

# Fit model for male professors only
male_data = data[data['gender'] == 'male']
X_male = sm.add_constant(male_data['beauty'])
y_male = male_data['score']

male_model = sm.OLS(y_male, X_male).fit()

# Equation of the line
slope = male_model.params[1]
intercept = male_model.params[0]
print(f"Equation of the line for male professors: y = {intercept:.2f} + {slope:.2f} * beauty")

# Visualization
plt.figure(figsize=(10, 6))
colors = {'male':'blue', 'female':'red'}

for gender, group_data in data.groupby('gender'):
    plt.scatter(group_data['beauty'], group_data['score'], color=colors[gender], label=gender)

# Add line for male professors
plt.plot(male_data['beauty'], male_model.predict(X_male), color='navy', linewidth=2, label='Male fit')
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 61 with threat_id: thread_WK6SbyndYTvaVu9ss49nLFLD
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Assuming your data is in a CSV file named 'professor_data.csv'
# Change the filename and path as necessary
data = pd.read_csv('professor_data.csv')

# Fit the multiple linear regression model
# Create dummy variables for gender if not already
data['gender'] = data['gender'].map({'female': 1, 'male': 0})  # Example mapping

X = data[['bty_avg', 'gender']]
y = data['score']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Display the summary of the regression
print(score_bty_gender_fit.summary())

# Interpretation
intercept = score_bty_gender_fit.params['const']  # Intercept
slope_bty = score_bty_gender_fit.params['bty_avg']  # Slope for beauty rating
slope_gender = score_bty_gender_fit.params['gender']  # Slope for gender

print(f"Intercept (const): {intercept}")
print(f"Slope for beauty rating (bty_avg): {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Create scatter plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, dodge=True, palette='Set1')
plt.title('Professor Evaluation Scores by Average Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 61 with threat_id: thread_BRplwcak1hLbOHxfdJk3YKuj
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# df = pd.read_csv('your_data.csv') # Uncomment and replace with your path

# Model fitting
df['intercept'] = 1  # Add an intercept term for the model
# Run OLS regression
model = sm.OLS(df['evaluation_score'], df[['intercept', 'beauty', 'gender_female']]).fit()  # assuming 'gender_female' is a binary variable for gender
print(model.summary())

# Equation extraction for male professors
# Assuming male is the reference category in regression
slope_beauty = model.params['beauty']
intercept = model.params['intercept']
print(f'Male Equation: score = {intercept} + {slope_beauty}*beauty')

# Check R-squared
print(f'R-squared: {model.rsquared}')

# Plotting the relationship
sns.lmplot(x='beauty', y='evaluation_score', hue='gender', data=df, markers=['o', 's'], palette='Set1')
plt.title('Relationship Between Beauty and Evaluation Score')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 62 with threat_id: thread_8Ls0FLdfFHfRkGBJz9ukdrKp
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data.csv')  # Uncomment and modify this line to load your dataset

# Example DataFrame setup (uncomment once your actual data is loaded)
# df = pd.DataFrame({
#     'score_avg': [4.5, 3.8, 4.0, 4.2],
#     'bty_avg': [3.5, 2.5, 4.0, 5.0],
#     'gender': ['Male', 'Female', 'Female', 'Male']
# })

# Prepare your data
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})  # Convert categorical variables to numerical
X = df[['bty_avg', 'gender']]
y = df['score_avg']

# Fit the multiple linear regression model
X = sm.add_constant(X)  # Adds a constant term to the predictor
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

# Interpretation
print(f'Intercept: {intercept}')
print(f'Slope of beauty rating: {slope_bty}')
print(f'Slope of gender: {slope_gender}')

# Scatter plot
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='bty_avg', y='score_avg', hue='gender', jitter=True, palette='Set1', alpha=0.7)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper right', labels=['Female', 'Male'])
plt.show()
##################################################
#Question 26.0, Round 62 with threat_id: thread_4wVQAri62Cvg6RSum8qUl99d
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample data for demonstration
data = {
    'beauty': [6, 8, 7, 9, 5, 7, 8, 5, 9, 6],  # beauty scores
    'score': [80, 90, 85, 95, 70, 75, 88, 70, 96, 82],  # evaluation scores
    'gender': ['M', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'M', 'F']  # gender of professors
}
df = pd.DataFrame(data)

# Create model with gender
model = sm.OLS.from_formula('score ~ beauty * gender', data=df).fit()

# Output summary
print(model.summary())
print(f'R-squared: {model.rsquared:.2f}')

# Plotting
plt.figure(figsize=(10, 5))
for gender, group in df.groupby('gender'):
    plt.scatter(group['beauty'], group['score'], label=gender, alpha=0.5)

# Adding regression lines
x = df['beauty']
y_male = model.predict(exog=dict(beauty=x, gender='M'))
y_female = model.predict(exog=dict(beauty=x, gender='F'))

plt.plot(x, y_male, color='blue', label='Male: y = 0.78 + 0.89x')
plt.plot(x, y_female, color='orange', label='Female: y = 0.36 + 0.56x')

plt.title("Beauty vs Evaluation Score by Gender")
plt.xlabel("Beauty Score")
plt.ylabel("Evaluation Score")
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 63 with threat_id: thread_hrBjOgjGyC1dIaEKFoK53gBS
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Assuming data is stored in a DataFrame named df with columns 'score', 'bty_avg', and 'gender'
# df = pd.read_csv('your_data.csv')  # Load your dataset (uncomment this line and provide your file)

# Example DataFrame structure:
# df = pd.DataFrame({
#     'score': [4.5, 4.0, 3.8, 4.2, 4.7],
#     'bty_avg': [3.2, 4.5, 2.5, 3.8, 4.8],
#     'gender': ['Male', 'Female', 'Female', 'Male', 'Female']
# })

# Transform gender to numeric
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['score']
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print model summary
print(score_bty_gender_fit.summary())

# Interpreting the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f"Intercept: {intercept}")
print(f"Slope for Average Beauty Rating (bty_avg): {slope_bty}")
print(f"Slope for Gender (1=Male, 0=Female): {slope_gender}")

# Create the scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, alpha=0.7)
plt.title('Average Professor Evaluation Score vs. Average Beauty Rating by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()
##################################################
#Question 26.0, Round 63 with threat_id: thread_OWYWtAJfLA8vVauKgvEnVwbB
import pandas as pd
import statsmodels.api as sm

# Sample data for demonstration (replace with your actual data)
data = {
    'beauty': [1, 2, 3, 4, 5],
    'score': [3, 4, 5, 7, 8],
    'gender': ['male', 'female', 'male', 'female', 'male']
}
df = pd.DataFrame(data)

# Encode gender
df['gender_encoded'] = df['gender'].map({'male': 0, 'female': 1})

# Define the independent variables with interaction
X = sm.add_constant(df[['beauty', 'gender_encoded', 'beauty * gender_encoded']])
y = df['score']

# Fit the model
model = sm.OLS(y, X).fit()

# Percent of variability explained
r_squared = model.rsquared * 100  # Percentage
print(f"Percent of variability explained by the model: {r_squared:.2f}%")

# Coefficients to create the equations
intercept = model.params['const']
beta_beauty = model.params['beauty']
beta_gender = model.params['gender_encoded']
beta_interaction = model.params['beauty * gender_encoded']

# Equation for male professors (gender_encoded = 0)
male_equation = f"Score = {intercept:.2f} + {beta_beauty:.2f} * Beauty"

print(f"Equation of the line (male professors): {male_equation}")

# Equation for female professors (gender_encoded = 1)
female_equation = f"Score = {intercept + beta_gender + beta_interaction:.2f} + {beta_beauty + beta_interaction:.2f} * Beauty"

print(f"Equation of the line (female professors): {female_equation}")

# Interpretation of the results
print("The relationship between beauty and evaluation score varies as follows:")
print(f"- For males: {male_equation}")
print(f"- For females: {female_equation}")
##################################################
#Question 26.1, Round 64 with threat_id: thread_Jq6rXMELtGQ616vFDWNboc9t
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_data.csv' with your actual file)
# df = pd.read_csv('your_data.csv')

# Example DataFrame structure
data = {
    'score_avg': [4.5, 3.9, 4.8, 4.2, 5.0, 3.5, 4.6, 3.8],
    'bty_avg': [7, 5, 9, 6, 10, 4, 8, 5],
    'gender': ['male', 'female', 'female', 'male', 'female', 'male', 'female', 'male']
}
df = pd.DataFrame(data)

# Convert categorical variable 'gender' into numeric variables
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Define the dependent variable and independent variables
X = df[['bty_avg', 'gender']]
y = df['score_avg']

# Add constant to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpreting the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope of Beauty Rating: {slope_bty}')
print(f'Slope of Gender: {slope_gender}')

# Jittered scatterplot
plt.figure(figsize=(8, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper right', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.0, Round 64 with threat_id: thread_Vhaky5ECdC7MnYPZLhCjuJWe
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame `df` with columns 'beauty', 'evaluation', and 'gender'
# Example: df = pd.read_csv('your_data.csv')

# Model fitting
X = df[['beauty', 'gender']]
X['gender'] = X['gender'].map({'male': 1, 'female': 0})  # Mapping gender to numeric
y = df['evaluation']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()

# Percent of variability explained
r_squared = model.rsquared * 100  # Convert to percentage

# Equation for male professors
male_coefficients = model.params[['const', 'beauty']]
male_equation = f"Score = {male_coefficients[0]:.2f} + {male_coefficients[1]:.2f} * Beauty"

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='beauty', y='evaluation', hue='gender', alpha=0.6)
sns.lineplot(data=df[df['gender'] == 'male'], 
             x='beauty',
             y=model.predict(X)[df['gender'] == 'male'],
             color='blue', label='Male Trendline')
sns.lineplot(data=df[df['gender'] == 'female'], 
             x='beauty',
             y=model.predict(X)[df['gender'] == 'female'],
             color='red', label='Female Trendline')

plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()

# Outputs
print(f"Percent of variability in score explained by the model: {r_squared:.2f}%")
print(f"Equation of the line for male professors: {male_equation}")
##################################################
#Question 26.1, Round 65 with threat_id: thread_M4Efn8lKwyJM5k8hnCtB1SXe
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame creation: replace this with your actual DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment this and load your data

# For demonstration purposes, let's assume the DataFrame is like this:
data = {
    'score_avg': [4.2, 3.8, 4.0, 4.5, 4.1, 3.9, 4.3, 4.7],
    'bty_avg': [3, 2, 4, 5, 3, 2, 4, 5],
    'gender': ['M', 'F', 'F', 'M', 'F', 'M', 'M', 'F']
}
df = pd.DataFrame(data)

# Convert gender to a categorical variable, which will be converted to dummy variables
df['gender'] = pd.Categorical(df['gender'])

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable to dummy variables
y = df['score_avg']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and the slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_M']  # Reference category is gender_F

print(f'Intercept: {intercept:.2f}')
print(f'Slope for Beauty Rating: {slope_bty:.2f}')
print(f'Slope for Gender (Male vs Female): {slope_gender:.2f}')

# Create scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, palette='pastel')
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 65 with threat_id: thread_2y3pVMQd81PLwwoOCAynlXDi
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('path_to_your_data.csv') # Uncomment and set your file path

# Model fitting for both genders
X = df[['beauty']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['evaluation_score']

# Fit model for both genders
model = sm.OLS(y, X).fit()
print(model.summary())

# Fit separate models for male and female professors
male_df = df[df['gender'] == 'male']
female_df = df[df['gender'] == 'female']

# Regression for male professors
X_male = male_df[['beauty']]
X_male = sm.add_constant(X_male)
y_male = male_df['evaluation_score']
model_male = sm.OLS(y_male, X_male).fit()

# Get the equation of the line for male professors
intercept_male = model_male.params[0]
slope_male = model_male.params[1]
print(f"Equation for male professors: y = {intercept_male} + {slope_male} * x")

# Visualizing the relationship
sns.lmplot(x='beauty', y='evaluation_score', hue='gender', data=df, aspect=2)
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.show()

# Variability explained by the model
r_squared = model.rsquared
print(f"Percentage of variability in score explained by the model: {r_squared * 100:.2f}%")
##################################################
#Question 26.1, Round 66 with threat_id: thread_FPgw33K5G4V9WgX2LYby9clz
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (assuming it's in a CSV file format)
# df = pd.read_csv('your_data_file.csv')  # Uncomment and provide your file path

# Fit the multiple linear regression model
# Assuming 'score_avg', 'bty_avg', and 'gender' columns are present in the dataframe
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical gender variable into dummy/indicator variables
y = df['score_avg']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary to interpret the results
print(score_bty_gender_fit.summary())

# Interpretation of the intercept and slopes
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]  # Adjust the index based on gender encoding

print(f"Intercept: {intercept}")
print(f"Slope for Beauty Rating: {slope_bty}")
print(f"Slope for Gender (encoded): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 66 with threat_id: thread_zhVHDyFj0wNbWcMFFqeGyecZ
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Sample Data - Replace with actual data
data = {
    'score': np.random.rand(100) * 100,  # evaluation scores
    'beauty': np.random.rand(100) * 10,  # beauty ratings
    'gender': np.random.choice(['male', 'female'], 100)  # gender
}
df = pd.DataFrame(data)

# Creating the model
df['gender_male'] = np.where(df['gender'] == 'male', 1, 0)
df['gender_female'] = np.where(df['gender'] == 'female', 1, 0)

# Fit the model - Beauty effect on scores based on gender
X = df[['beauty', 'gender_male', 'gender_female']]
X = sm.add_constant(X)  # adding a constant
y = df['score']

model = sm.OLS(y, X).fit()

# Get R-squared
r_squared = model.rsquared
percent_variability_explained = r_squared * 100

# Coefficients for male professors
male_coeffs = model.params[['const', 'beauty', 'gender_male']]
equation_male = f'Score = {male_coeffs["const"]} + {male_coeffs["beauty"]} * Beauty'

# Summary of the model
print(model.summary())

# Print results
print(f"Percent of variability explained by the model: {percent_variability_explained:.2f}%")
print(f"Equation of the line for male professors: {equation_male}")
##################################################
#Question 26.1, Round 67 with threat_id: thread_8yXaEkbXOcKcOPJ9CkKBW4YL
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Assuming the dataset is in a CSV format and has the necessary columns.
# Replace 'your_dataset.csv' with the path to your actual data file.
data = pd.read_csv('your_dataset.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = data['score']  # Adjust 'score' if your column name is different
X = sm.add_constant(X)  # Add constant term for intercept
model = sm.OLS(y, X).fit()
score_bty_gender_fit = model

# Print model summary to interpret coefficients
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = model.params[0]
slope_bty = model.params[1]
slope_gender = model.params[2]

print(f"Intercept: {intercept}")
print(f"Slope of beauty rating: {slope_bty}")
print(f"Slope of gender (if male=1): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', data=data, jitter=True, hue='gender', palette='Set1', dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 67 with threat_id: thread_pCDz7VlHVWYJuTdghc7D8nfq
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your data here
# df = pd.read_csv('your_data.csv')

# For demonstration, let's create a hypothetical dataset
data = {
    'beauty': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'evaluation_score': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'gender': ['male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'male'],
}
df = pd.DataFrame(data)

# Model fitting for overall dataset
X = df[['beauty']]
y = df['evaluation_score']
X = sm.add_constant(X) # Adding the intercept

model = sm.OLS(y, X).fit()
print("Overall Model Summary:")
print(model.summary())

# Percent of variability explained (R-squared)
percent_variability = model.rsquared * 100

# Model fitting for just male professors
df_male = df[df['gender'] == 'male']
X_male = df_male[['beauty']]
y_male = df_male['evaluation_score']
X_male = sm.add_constant(X_male)  # Adding the intercept

model_male = sm.OLS(y_male, X_male).fit()
equation_male = f"Evaluation Score = {model_male.params[0]:.2f} + {model_male.params[1]:.2f} * Beauty"

# Plotting the relationship
plt.figure(figsize=(10, 6))
plt.scatter(df_male['beauty'], df_male['evaluation_score'], color='blue', label='Male professors')
plt.scatter(df[df['gender'] == 'female']['beauty'], df[df['gender'] == 'female']['evaluation_score'], color='orange', label='Female professors')

# Line of best fit for Male professors
plt.plot(df_male['beauty'], model_male.predict(X_male), color='blue')

plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Relationship between Beauty and Evaluation Score')
plt.legend()
plt.show()

# Display results
print(f"Percent of variability explained by the model: {percent_variability:.2f}%")
print(f"Equation of the line for male professors: {equation_male}")
##################################################
#Question 26.1, Round 68 with threat_id: thread_cqYqLAbnH0PxkTnH6JLW6Hip
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame creation (Replace this with your data loading)
# df = pd.read_csv('your_data.csv')  # Load your dataset here

# Assuming df is your DataFrame and it includes 'score_avg', 'bty_avg', and 'gender' columns

# Preparing the data
X = df[['bty_avg', 'gender']]
X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})  # Convert gender to a numerical format if needed
y = df['score_avg']

# Adding constant for intercept
X = sm.add_constant(X)

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Model summary
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_beauty}')
print(f'Slope for Gender: {slope_gender}')

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, palette='coolwarm', alpha=0.7)
plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 68 with threat_id: thread_NqgblgJYa3D9CdcqT8SqRhJ7
import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('your_data.csv')

# Filter data for male and female professors
data_male = data[data['gender'] == 'Male']
data_female = data[data['gender'] == 'Female']

# Fit the model for male professors
X_male = sm.add_constant(data_male['beauty'])  # Add a constant for intercept
y_male = data_male['evaluation_score']
model_male = sm.OLS(y_male, X_male).fit()

# Fit the model for female professors
X_female = sm.add_constant(data_female['beauty'])  # Add a constant for intercept
y_female = data_female['evaluation_score']
model_female = sm.OLS(y_female, X_female).fit()

# Print the summaries to get R-squared values and coefficients
print("Male Model Summary:")
print(model_male.summary())
print("\nFemale Model Summary:")
print(model_female.summary())

# Extract results for equations
equation_male = f'y = {model_male.params[0]:.2f} + {model_male.params[1]:.2f} * beauty'
equation_female = f'y = {model_female.params[0]:.2f} + {model_female.params[1]:.2f} * beauty'

print(f"\nEquation for Male Professors: {equation_male}")
print(f"Equation for Female Professors: {equation_female}")

# Visualizing the relationship
sns.scatterplot(x=data['beauty'], y=data['evaluation_score'], hue=data['gender'])
plt.plot(data_male['beauty'], model_male.predict(X_male), color='blue', label='Male Fit')
plt.plot(data_female['beauty'], model_female.predict(X_female), color='orange', label='Female Fit')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.title('Beauty vs Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 69 with threat_id: thread_1wJWFVswpXIroMXHpEt1aWEs
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data loading (modify this line to load your actual dataset)
# data = pd.read_csv('your_data.csv')

# Assuming your DataFrame is called `data`
# and includes 'score', 'bty_avg', and 'gender' columns.

# Encoding gender as numerical values
data['gender_encoded'] = data['gender'].map({'male': 1, 'female': 0})

# Preparing the model
X = data[['bty_avg', 'gender_encoded']]
y = data['score']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Displaying the model summary
print(score_bty_gender_fit.summary())

# Interpreting the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_encoded']

print(f'Intercept (constant): {intercept}')
print(f'Slope (beauty rating): {slope_beauty}')
print(f'Slope (gender): {slope_gender}')

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', data=data, jitter=True, hue='gender', dodge=True, palette='Set2')
plt.title('Professor Evaluation Score vs Beauty Rating by Gender')
plt.xlabel('Average Beauty Rating (bty_avg)')
plt.ylabel('Average Professor Evaluation Score (score)')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 69 with threat_id: thread_vB7mhhspqBBACmAKpdJeNp4G
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame creation (an example, replace with actual data)
# Assuming 'data' is a DataFrame containing the columns: 'score', 'beauty', 'gender'
data = pd.DataFrame({
    'score': np.random.rand(100) * 5, # Example evaluation scores
    'beauty': np.random.rand(100) * 10, # Example beauty scores
    'gender': np.random.choice(['male', 'female'], size=100) # Example gender
})

# Fit the model
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True) # Convert gender to numeric
y = data['score']
X = sm.add_constant(X) # Adding a constant

model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Predicting evaluation scores
data['predicted_score'] = model.predict(X)

# Plotting the relationship between beauty and score
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['beauty'], y=data['score'], hue=data['gender'])
plt.axline((0, 3.23), slope=0.67, color='blue', label='Male Fit Line')
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 70 with threat_id: thread_Sq7H52LAzQ8AaurwdHF8Tk4o
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import jitter

# Load your dataset (replace 'your_data.csv' with the path to your dataset)
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = data['avg_prof_eval_score']
X = sm.add_constant(X)  # Add a constant term to the predictor

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_male']  # Assuming 'gender' is binary: female as 0, male as 1

print(f"Intercept (constant term): {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender (male): {slope_gender}")

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='avg_prof_eval_score', hue='gender', data=data, jitter=True, palette="Set2")
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 70 with threat_id: thread_fJ4pfpXdHllc3WiPvtsEmtE4
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample data setup
# Assume the data has columns 'score', 'beauty', and 'gender'
data = pd.DataFrame({
    'score': [80, 85, 90, 92, 78, 75, 88, 90],
    'beauty': [5, 7, 8, 9, 4, 3, 6, 10],
    'gender': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'female']
})

# Fit the model (assuming you have numerical data for beauty and scores)
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable
y = data['score']
X = sm.add_constant(X)  # Adding a constant term for the intercept

model = sm.OLS(y, X).fit()
print(model.summary())

# Extract R-squared value for variability explanation
r_squared = model.rsquared
print(f"Percent of variability explained by the model: {r_squared * 100:.2f}%")

# Equation for just male professors
male_profs = data[data['gender'] == 'male']
X_male = male_profs[['beauty']]
X_male = sm.add_constant(X_male)  # Adding a constant term
model_male = sm.OLS(male_profs['score'], X_male).fit()
print(f"Equation for Male Professors: score = {model_male.params[0]:.2f} + {model_male.params[1]:.2f} * beauty")

# Plotting
plt.figure(figsize=(10, 6))
for gender in ['male', 'female']:
    subset = data[data['gender'] == gender]
    plt.scatter(subset['beauty'], subset['score'], label=gender)
    plt.plot(subset['beauty'], model.predict(X.loc[subset.index]), label=f'Fit - {gender}', linestyle='--')

plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Relationship between Beauty and Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 71 with threat_id: thread_zwXx9gj7IuApuEsL40AUk5O8
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame creation (replace this with your actual data file loading)
# data = pd.read_csv('your_data_file.csv') # Uncomment this and load your actual dataset
# Example DataFrame for demonstration purposes
data = pd.DataFrame({
    'score_avg': np.random.uniform(1, 5, 100),
    'bty_avg': np.random.uniform(1, 5, 100),
    'gender': np.random.choice(['Male', 'Female'], 100)
})

# Encoding gender as numerical values (Male: 0, Female: 1)
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# Fitting the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = sm.add_constant(X)  # Adding a constant (intercept)
y = data['score_avg']
score_bty_gender_fit = sm.OLS(y, X).fit()

# Model summary
print(score_bty_gender_fit.summary())

# Interpret coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope of beauty rating: {slope_bty}')
print(f'Slope of gender: {slope_gender}')

# Creating a scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True)
plt.title('Scatterplot of Professor Evaluation Score by Beauty Rating (Colored by Gender)')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 71 with threat_id: thread_y3nufYE4mIAF3kpqnJUfSDe6
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load your dataset
# data = pd.read_csv('your_data.csv')  # Uncomment and modify this to load your dataset

# Example DataFrame creation for demonstration
data = pd.DataFrame({
    'beauty': np.random.rand(100),
    'score': np.random.rand(100) * 100,
    'gender': np.random.choice(['male', 'female'], 100)
})

# Fit the model
model = sm.OLS(data['score'], sm.add_constant(data[['beauty', 'gender']].replace({'gender': {'male': 1, 'female': 0}}))).fit()

# Variability explained
r_squared = model.rsquared * 100  # percent
print(f"Percent of variability in score explained by the model: {r_squared:.2f}%")

# Equation for male professors (gender = 1)
intercept = model.params['const']
beauty_coef = model.params['beauty']
male_equation = f"Score = {intercept:.2f} + {beauty_coef:.2f} * Beauty"
print(f"Equation of the line for just male professors: {male_equation}")

# Analyze the relationship for male and female
data['predicted_score'] = model.predict(sm.add_constant(data[['beauty', 'gender']].replace({'gender': {'male': 1, 'female': 0}})))
male_data = data[data['gender'] == 'male']
female_data = data[data['gender'] == 'female']

# Correlation to see relationship changes
male_corr = male_data['beauty'].corr(male_data['predicted_score'])
female_corr = female_data['beauty'].corr(female_data['predicted_score'])

print(f"Correlation (Beauty vs Score) for Male Professors: {male_corr:.2f}")
print(f"Correlation (Beauty vs Score) for Female Professors: {female_corr:.2f}")
##################################################
#Question 26.1, Round 72 with threat_id: thread_yqFCCFLL8OIt5FidSF3XLKP7
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Assuming data is in a CSV, replace 'your_data.csv' with the correct filename
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
# Define the independent variables and the dependent variable
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = data['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'gender_Male' is created by pd.get_dummies()

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating (bty_avg): {slope_bty}")
print(f"Slope for gender (Male): {slope_gender}")

# Create a scatter plot with jitter
sns.scatterplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True)
plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating (bty_avg)')
plt.ylabel('Average Professor Evaluation Score (score)')
plt.show()
##################################################
#Question 26.0, Round 72 with threat_id: thread_BGWVPvixPzAHTKWrNkrErP5Y
import pandas as pd
import statsmodels.api as sm

# Load your data
data = pd.read_csv('your_data_file.csv')  # Ensure to replace with your actual data file

# Fit the linear regression model for beauty and gender
model = sm.OLS(data['evaluation_score'], sm.add_constant(data[['beauty', 'gender']])).fit()
print(model.summary())

# Extracting R-squared value (percent variability explained)
r_squared = model.rsquared
print(f"Percentage of Variability Explained: {r_squared * 100:.2f}%")

# Equation for just male professors
male_data = data[data['gender'] == 'male']
male_model = sm.OLS(male_data['evaluation_score'], sm.add_constant(male_data['beauty'])).fit()
intercept_male = male_model.params[0]
slope_male = male_model.params[1]
print(f"Equation for Male Professors: Score = {intercept_male:.2f} + {slope_male:.2f} * Beauty")

# Fit the model for female professors for comparison
female_data = data[data['gender'] == 'female']
female_model = sm.OLS(female_data['evaluation_score'], sm.add_constant(female_data['beauty'])).fit()
intercept_female = female_model.params[0]
slope_female = female_model.params[1]
print(f"Equation for Female Professors: Score = {intercept_female:.2f} + {slope_female:.2f} * Beauty")

# Comparing relationships
print(f"Slope for Male Professors: {slope_male:.2f}, Slope for Female Professors: {slope_female:.2f}")
##################################################
#Question 26.1, Round 73 with threat_id: thread_JxefO5BEXLHzI0CiaLIPVQnJ
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_data.csv' with your actual data file)
# df = pd.read_csv('your_data.csv')

# Example data frame
# Assuming df has columns: 'score', 'bty_avg', 'gender'
# Uncomment and replace the above line with actual data loading

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy/indicator variables
Y = df['score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the model
score_bty_gender_fit = sm.OLS(Y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_M']  # Assuming 'gender_M' is the dummy variable for Male

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender (Male): {slope_gender}")

# Scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, palette='Set2', alpha=0.6)
plt.title('Average Professor Evaluation Score by Average Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 73 with threat_id: thread_qjwZCZQzt0Aib9YoAeK1VyTS
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
# Assuming `data` is a DataFrame that contains columns 'beauty', 'score', and 'gender'
data = pd.read_csv("your_data.csv")  # Replace with your actual data file

# Create model
def fit_model(data):
    # Encode gender as dummy variables
    data['is_male'] = np.where(data['gender'] == 'male', 1, 0)
    
    # Fit the model
    X = data[['beauty', 'is_male']]
    y = data['score']
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    
    # Get R-squared value
    r_squared = model.rsquared
    
    # Get coefficients for interpretation
    coeffs = model.params
    intercept = coeffs[0]
    beauty_coeff = coeffs[1]
    gender_coeff = coeffs[2]
    
    return r_squared, intercept, beauty_coeff, gender_coeff

# Fit the model
r_squared, intercept, beauty_coeff, gender_coeff = fit_model(data)

# Print results
print(f"R^2: {r_squared * 100:.2f}%")
print(f"Equation of the line for male professors: score = {intercept} + {beauty_coeff} * beauty + {gender_coeff}")

# Compare the intercept (for male) and calculate for female
female_intercept = intercept + gender_coeff
print(f"Equation of the line for female professors: score = {female_intercept} + {beauty_coeff} * beauty")

# Visualize the relationship
plt.scatter(data['beauty'], data['score'], c=data['is_male'], cmap='coolwarm', label='Gender')
plt.plot(data['beauty'], intercept + beauty_coeff * data['beauty'], color='blue', label='Male Fit')
plt.plot(data['beauty'], female_intercept + beauty_coeff * data['beauty'], color='red', label='Female Fit')
plt.title('Beauty vs. Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 74 with threat_id: thread_nMkpDWKD5nwwhcimHuMABhKS
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Assuming the data is in a CSV file named 'professor_evaluation_data.csv'
# df = pd.read_csv('professor_evaluation_data.csv')

# Sample DataFrame creation, replace it with your actual data loading method
data = {
    'avg_prof_eval_score': [3.5, 4.0, 4.2, 2.9, 3.8, 4.5],
    'avg_beauty_rating': [3.0, 4.0, 4.5, 2.0, 3.5, 4.8],
    'gender': ['M', 'F', 'F', 'M', 'F', 'M']
}
df = pd.DataFrame(data)

# Convert gender to numerical (0 for Male, 1 for Female)
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# Define the independent variables (X) and the dependent variable (y)
X = df[['avg_beauty_rating', 'gender']]
y = df['avg_prof_eval_score']

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Fit the Multiple Linear Regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the regression results
print(score_bty_gender_fit.summary())

# Interpret the model
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['avg_beauty_rating']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept (constant): {intercept}')
print(f'Slope (beauty rating): {slope_beauty}')
print(f'Slope (gender): {slope_gender}')

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_beauty_rating', y='avg_prof_eval_score', hue='gender', jitter=True, palette='deep')
plt.title('Scatter Plot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Male', 'Female'])
plt.show()
##################################################
#Question 26.0, Round 74 with threat_id: thread_1sb8ZaHKZ1GDsh1KqTxtxGDj
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Make sure to replace 'your_data.csv' with the path to your dataset
data = pd.read_csv('your_data.csv')

# Fit the model score_bty_gender_fit
# Assuming 'beauty' is the beauty score, 'evaluation' is the evaluation score, and 'gender' is the gender of professors
X = pd.get_dummies(data[['beauty', 'gender']], drop_first=True)
y = data['evaluation']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Get the percentage of variability explained (R-squared)
r_squared = model.rsquared * 100  # Multiply by 100 for percentage
print(f"Percentage of variability in score explained by the model: {r_squared:.2f}%")

# Get the equation of the line for just male professors
coefficients = model.params
male_eq = f"Evaluation Score = {coefficients['const']:.2f} + {coefficients['beauty']:.2f} * Beauty Score + {coefficients['gender_male']:.2f} * 1"
print(f"Equation for male professors: {male_eq}")

# Plotting the relationship
sns.lmplot(x='beauty', y='evaluation', hue='gender', data=data, markers=["o", "s"], palette="muted")
plt.title('Relationship between Beauty and Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 75 with threat_id: thread_c2ytryRep6j4A0yzRDV6pwxV
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('your_data_file.csv') # Uncomment and replace with your actual file

# Assuming your DataFrame is called df and it contains the necessary columns
# df should have columns: 'score_avg', 'bty_avg', 'gender'

# Fit multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True) # Convert categorical gender to dummy variables
y = df['score_avg']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fitting the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Output the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]  # for bty_avg
slope_gender = score_bty_gender_fit.params[2]  # for gender (1 if male, 0 if female)

print(f"Intercept: {intercept}")
print(f"Slope of Beauty Rating: {slope_bty}")
print(f"Slope of Gender (male): {slope_gender}")

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper left')
plt.show()
##################################################
#Question 26.0, Round 75 with threat_id: thread_wvbE2frZuhrQiRaH395SOCWG
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data loading (Replace with your data loading method)
# df = pd.read_csv('your_data.csv')

# Fit the model
model = sm.OLS(df['score'], sm.add_constant(df[['beauty', 'gender']])).fit()

# 1. Percent of variability explained (R-squared)
r_squared = model.rsquared
percentage_variability_explained = r_squared * 100

# 2. Equation of the line for just male professors
male_professors = df[df['gender'] == 'male']
male_model = sm.OLS(male_professors['score'], sm.add_constant(male_professors['beauty'])).fit()
intercept = male_model.params['const']
slope = male_model.params['beauty']
equation_male = f"score = {intercept:.2f} + {slope:.2f} * beauty"

# 3. Relationship visualization between beauty and evaluation score by gender
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='beauty', y='score', hue='gender', alpha=0.6)
x_vals = np.linspace(df['beauty'].min(), df['beauty'].max(), 100)

# Adding regression lines for each gender
for gender, color in zip(['male', 'female'], ['blue', 'orange']):
    gender_model = sm.OLS(df[df['gender'] == gender]['score'], sm.add_constant(df[df['gender'] == gender]['beauty'])).fit()
    plt.plot(x_vals, gender_model.predict(sm.add_constant(x_vals)), color=color, label=f'Regression line for {gender}')

plt.title('Beauty vs. Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.legend()
plt.show()

# Output results
results = {
    "percentage_variability_explained": percentage_variability_explained,
    "equation_male": equation_male
}

print(results)
##################################################
#Question 26.1, Round 76 with threat_id: thread_OvlOZJtK3FnwYc0ZNccPzL5O
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame 'df' with 'score', 'bty_avg', and 'gender' columns

# Fit multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # One-hot encode the gender variable
y = df['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)
score_bty_gender_fit = sm.OLS(y, X).fit()

# Output the summary of the model
print(score_bty_gender_fit.summary())

# Coefficients interpretation
intercept = score_bty_gender_fit.params['const']
beauty_slope = score_bty_gender_fit.params['bty_avg']
gender_slope = score_bty_gender_fit.params['gender_male']  # Assuming 'gender_male' is the encoded male variable

print(f"Intercept: {intercept}")
print(f"Slope of beauty rating: {beauty_slope}")
print(f"Slope of gender (male): {gender_slope}")

# Jittered scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True)
plt.title('Jittered Scatterplot of Score by Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 76 with threat_id: thread_NifTDENaTsOPuGBN4RbQ6PVo
import pandas as pd
import statsmodels.api as sm

# Load your data
# Replace 'your_data.csv' with the path to your dataset
data = pd.read_csv('your_data.csv')  # Ensure your CSV has columns 'Score', 'Beauty', and 'Gender'

# Fit the model: Score ~ Beauty + Gender
X = data[['Beauty', 'Gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable Gender into dummy/indicator variables
y = data['Score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()

# Summary of the model
summary = model.summary()
rsquared = model.rsquared

# Filter the data for male professors
male_data = data[data['Gender'] == 'Male']
male_X = sm.add_constant(male_data['Beauty'])
male_model = sm.OLS(male_data['Score'], male_X).fit()

# Get the equation of the line for male professors
male_slope = male_model.params[1]
male_intercept = male_model.params[0]
equation_male = f"Score = {male_intercept:.2f} + {male_slope:.2f} * Beauty"

# Relationship evaluation
gender_effect = model.params

# Results
print(f"R-squared: {rsquared:.3f}")  # Percent of variability explained by the model
print(f"Equation of the line for male professors: {equation_male}")
print("Gender effect on the relationship:")
print(gender_effect)  # Show effect of each gender

# You can adjust the print statements according to your requirements.
##################################################
#Question 26.1, Round 77 with threat_id: thread_GmMKE4XDxDqKex0RcW8gbLqO
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into a DataFrame
# df = pd.read_csv('your_data_file.csv')  # Uncomment and replace with your data file

# Assuming your DataFrame contains columns 'score', 'bty_avg', and 'gender'
# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Save the fitted model
score_bty_gender_fit = model

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpreting the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_M']  # Assuming 'M' is one of the dummy variables for gender

print(f'Intercept: {intercept}')
print(f'Slope of beauty rating: {slope_bty}')
print(f'Slope of gender: {slope_gender}')  # Must interpret based on the model context

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, palette='Set1')
plt.title('Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 77 with threat_id: thread_JadSqxwfDRKKFF3XhnF66K6F
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_data.csv')  # Load your actual dataset here

# Assuming your dataframe has 'beauty_score', 'evaluation_score', and 'gender' columns
# Prepare the data
df_male = df[df['gender'] == 'male']
df_female = df[df['gender'] == 'female']

# Model for male professors
X_male = sm.add_constant(df_male['beauty_score'])  # Add constant (intercept)
y_male = df_male['evaluation_score']
model_male = sm.OLS(y_male, X_male).fit()

# For female professors
X_female = sm.add_constant(df_female['beauty_score'])  # Add constant (intercept)
y_female = df_female['evaluation_score']
model_female = sm.OLS(y_female, X_female).fit()

# Output the results
print(f"Percentage of Variability Explained (Male): {model_male.rsquared * 100:.2f}%")
print(f"Equation for Male Professors: y = {model_male.params[0]:.2f} + {model_male.params[1]:.2f} * x")

print(f"Percentage of Variability Explained (Female): {model_female.rsquared * 100:.2f}%")
print(f"Equation for Female Professors: y = {model_female.params[0]:.2f} + {model_female.params[1]:.2f} * x")

# Visualizing the relationship
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='beauty_score', y='evaluation_score', hue='gender')
plt.plot(df_male['beauty_score'], model_male.predict(X_male), color='blue')
plt.plot(df_female['beauty_score'], model_female.predict(X_female), color='red')
plt.title('Beauty Score vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 78 with threat_id: thread_Te8lSFVu1bU5T9ZsolZ341Y0
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_file.csv' with the path to your actual dataset
data = pd.read_csv('your_file.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variables
y = data['score']  # Assuming the score column is named 'score'

# Add a constant to the model for the intercept
X = sm.add_constant(X)

# Fit the regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming Male is coded as '1'

print(f"Intercept (expected score when bty_avg is 0 and gender is female): {intercept}")
print(f"Slope for beauty rating (change in score per unit increase in bty_avg): {slope_bty}")
print(f"Slope for gender (difference in score between males and females): {slope_gender}")

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', data=data, hue='gender', jitter=True, dodge=True, alpha=0.7)
plt.title('Scatterplot of Score by Average Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 78 with threat_id: thread_W5Va3IU6MMmQML1QXZ9TiZJ2
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('your_data.csv') # Specify your dataset file

# Assume 'beauty' is your independent variable and 'evaluation_score' your dependent variable

# Fit models for male and female professors
model_male = sm.OLS(df[df['gender'] == 'male']['evaluation_score'], 
                    sm.add_constant(df[df['gender'] == 'male']['beauty'])).fit()

model_female = sm.OLS(df[df['gender'] == 'female']['evaluation_score'], 
                      sm.add_constant(df[df['gender'] == 'female']['beauty'])).fit()

# Extract R-squared value for the variability explained
r_squared = model_male.rsquared * 100  # Convert to percentage
print(f"Percent of variability explained by the model for male professors: {r_squared:.2f}%")

# Output model summary (equation of the line)
print("Equation for male professors: y =", model_male.params[0], "+", model_male.params[1], "* x")

# If needed, also check the female model
print("Equation for female professors: y =", model_female.params[0], "+", model_female.params[1], "* x")

# Plotting the relationships
plt.scatter(df[df['gender'] == 'male']['beauty'], df[df['gender'] == 'male']['evaluation_score'], color='blue', label='Male Professors')
plt.scatter(df[df['gender'] == 'female']['beauty'], df[df['gender'] == 'female']['evaluation_score'], color='red', label='Female Professors')

# Plotting regression lines
plt.plot(df[df['gender'] == 'male']['beauty'], model_male.predict(sm.add_constant(df[df['gender'] == 'male']['beauty'])), color='blue', linestyle='--')
plt.plot(df[df['gender'] == 'female']['beauty'], model_female.predict(sm.add_constant(df[df['gender'] == 'female']['beauty'])), color='red', linestyle='--')

plt.title('Beauty vs Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 79 with threat_id: thread_sZfLCjLs9t4Gb68tESc95qMm
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your dataset
# df = pd.read_csv('your_file.csv')  # Replace 'your_file.csv' with your dataset

# Assuming df has 'score', 'bty_avg', and 'gender' columns
# Preparing the data for regression
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']

# Fit the multiple linear regression model
model = sm.OLS(y, sm.add_constant(X)).fit()
score_bty_gender_fit = model

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Create scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Scatter Plot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', loc='upper left')
plt.show()
##################################################
#Question 26.0, Round 79 with threat_id: thread_4NTDkRNmgtiD260wmVdcLubr
import pandas as pd
import statsmodels.api as sm

# Sample data for demonstration purposes
# In a real scenario, load your dataset instead
data = {
    'beauty': [2, 3, 5, 4, 1, 4, 5, 3, 2, 4],
    'evaluation_score': [2.5, 3.3, 4.7, 4.2, 1.9, 4.5, 5.0, 3.8, 2.1, 4.0],
    'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female']
}

df = pd.DataFrame(data)

# Defining the model
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, columns=['gender'], drop_first=True)  # Using dummies for gender
y = df['evaluation_score']

# Fitting the model
model = sm.OLS(y, sm.add_constant(X)).fit()

# Variability explained
explained_variance = model.rsquared * 100 # convert to percent

# Extracting the equation for male professors
coefficients = model.params
intercept = coefficients['const']
beta_beauty = coefficients['beauty']
beta_gender_male = coefficients['gender_male'] if 'gender_male' in coefficients else 0

# Equation for male professors
equation_male = f"y = {intercept} + {beta_beauty}(beauty) + {beta_gender_male}(gender_male)"

# Analyzing the relationship
summary = model.summary()
relationship = summary.tables[1]  # Summary table with coefficients

print(f"Percent of variability explained by the model: {explained_variance:.2f}%")
print(f"Equation of the line for male professors: {equation_male}")
print("Relationship between beauty and evaluation score:")
print(relationship)

# For further diagnostics on difference by gender, consider using:
import matplotlib.pyplot as plt
import seaborn as sns

sns.lmplot(data=df, x='beauty', y='evaluation_score', hue='gender', palette='Set1', aspect=1.5)
plt.title('Relationship between Beauty and Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 80 with threat_id: thread_AkpXES6o323kcubNydEsrafG
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data loading (replace with your DataFrame)
# df = pd.read_csv('your_data_file.csv')

# Assuming your DataFrame looks something like this:
# df = pd.DataFrame({
#     'score_avg': [3.5, 4.0, 3.8, 4.2],
#     'bty_avg': [4.0, 3.5, 4.5, 4.2],
#     'gender': ['Male', 'Female', 'Female', 'Male']
# })

# Convert gender to a numeric variable: Male = 0, Female = 1
df['gender_numeric'] = df['gender'].map({'Male': 0, 'Female': 1})

# Define the independent variables and the dependent variable
X = df[['bty_avg', 'gender_numeric']]
y = df['score_avg']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret the parameters
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_numeric']

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', data=df, hue='gender', jitter=True, palette='Set2', dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 80 with threat_id: thread_rxXMy68xZ0g6Qjq97n8V2vae
import pandas as pd
import statsmodels.api as sm

# Load your data
# Assuming data is in a CSV with columns 'Beauty', 'Score', and 'Gender'
data = pd.read_csv('your_data_file.csv')

# Separate data for male and female professors
data_male = data[data['Gender'] == 'Male']
data_female = data[data['Gender'] == 'Female']

# Fit model for Male
X_male = sm.add_constant(data_male['Beauty'])  # adding a constant
model_male = sm.OLS(data_male['Score'], X_male).fit()

# Fit model for Female
X_female = sm.add_constant(data_female['Beauty'])  # adding a constant
model_female = sm.OLS(data_female['Score'], X_female).fit()

# Print summary statistics
print("Male Model Summary:")
print(model_male.summary())
print("\nFemale Model Summary:")
print(model_female.summary())

# Display equations
print(f'Male Equation: Score = {model_male.params[0]:.2f} + {model_male.params[1]:.2f} * Beauty')
print(f'Female Equation: Score = {model_female.params[0]:.2f} + {model_female.params[1]:.2f} * Beauty')

# Display R-squared (percent variability explained)
print(f'Male R-squared: {model_male.rsquared * 100:.2f}%')
print(f'Female R-squared: {model_female.rsquared * 100:.2f}%')
##################################################
#Question 26.1, Round 81 with threat_id: thread_M4pLPrhM2M2toEjO7rY1jOnM
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data; replace 'your_data.csv' with your actual file's name
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
# Assuming 'score_avg' is the average professor evaluation score,
# 'bty_avg' is the average beauty rating, and 'gender' is a binary categorical variable
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender into dummy variables
y = data['score_avg']
X = sm.add_constant(X)  # Add intercept term

# Create the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpret intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'gender_Male' is a dummy variable for Male

print(f'Intercept: {intercept}')
print(f'Slope (Beauty Rating): {slope_bty}')
print(f'Slope (Gender): {slope_gender}')

# Scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', data=data, hue='gender', jitter=True, palette='Set1')
plt.title('Scatterplot of Average Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 81 with threat_id: thread_zJdKGclN4mL0gEtr2qp3xsfe
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load your data into a DataFrame (assuming a CSV file format)
# df = pd.read_csv('your_file.csv')

# Example data structure (you will need to adjust it based on your actual data)
# df = pd.DataFrame({
#     'evaluation_score': [/* your scores */],
#     'beauty_score': [/* your beauty scores */],
#     'gender': [/* 'male' or 'female' */]
# })

# Creating the model: response variable -> evaluation_score; predictors -> beauty_score and gender
# Convert gender to a numerical variable: male = 1, female = 0
df['gender_numeric'] = np.where(df['gender'] == 'male', 1, 0)

# Fit the model
X = df[['beauty_score', 'gender_numeric']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['evaluation_score']

model = sm.OLS(y, X).fit()

# Print model summary to get R-squared which indicates variability explained by the model
print(model.summary())

# Get the equation for male professors
# Assuming 'gender' = 1 for male: y = b0 + b1 * beauty_score
b0, b1 = model.params
equation_male = f"Score = {b0} + {b1} * Beauty_Score"

print("Equation for Male Professors:", equation_male)

# Analyze the impact of gender on relationship with beauty score
# You can compare the adjusted model for males and females separately if needed
df_male = df[df['gender'] == 'male']
df_female = df[df['gender'] == 'female']

model_male = sm.OLS(df_male['evaluation_score'], sm.add_constant(df_male['beauty_score'])).fit()
model_female = sm.OLS(df_female['evaluation_score'], sm.add_constant(df_female['beauty_score'])).fit()

print("Male Model Summary:\n", model_male.summary())
print("Female Model Summary:\n", model_female.summary())
##################################################
#Question 26.1, Round 82 with threat_id: thread_dZrlnxgNzhJfx3NBhEnYTQCh
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample DataFrame creation - replace this with your actual data
# df = pd.read_csv('your_data.csv')  # Uncomment and modify to load your data
data = {
    'score_avg': [4.2, 3.8, 4.5, 4.0, 3.9],
    'bty_avg': [4.5, 3.5, 5.0, 4.0, 4.2],
    'gender': ['male', 'female', 'female', 'male', 'female']
}
df = pd.DataFrame(data)

# Convert categorical variable 'gender' to numerical format
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
y = df['score_avg']
X = sm.add_constant(X)  # Adds the intercept term
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, palette='Set1')
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Male', 'Female'], loc='upper left')
plt.show()
##################################################
#Question 26.0, Round 82 with threat_id: thread_ynBUDYMDZ5Z5k0ccC853B5OZ
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load your data
# Assuming you have a dataframe 'df' with columns 'beauty', 'evaluation', and 'gender'
# df = pd.read_csv('your_data.csv')

# Example data structure: Replace with your actual data loading method
# df = pd.DataFrame({
#     'beauty': np.random.rand(100)*10,
#     'evaluation': np.random.rand(100)*10,
#     'gender': np.random.choice(['male', 'female'], 100)
# })

# Fit the model
df['intercept'] = 1
model = sm.OLS(df['evaluation'], df[['intercept', 'beauty', 'gender_male']])  # Use gender encoding
results = model.fit()

# Variability explained
r_squared = results.rsquared
print(f'R-squared value: {r_squared:.4f}')

# Equation for male professors
male_results = sm.OLS(df[df['gender'] == 'male']['evaluation'], 
                      df.loc[df['gender'] == 'male', ['intercept', 'beauty']]).fit()

male_coefficients = male_results.params
print(f'Equation for male professors: evaluation = {male_coefficients["intercept"]:.4f} + {male_coefficients["beauty"]:.4f} * beauty')

# Relationship comparison
fig, ax = plt.subplots()
for key, grp in df.groupby(['gender']):
    ax = grp.plot(ax=ax, kind='scatter', x='beauty', y='evaluation', label=key)
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 83 with threat_id: thread_SsVDCv2may1I4iBDby3iCrt4
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Assuming the data is loaded into a DataFrame named df
# df = pd.read_csv('your_data_file.csv')

# Example DataFrame structure
# df = pd.DataFrame({
#     'score_avg': [4.2, 4.5, 3.8, 4.0, ...],
#     'bty_avg': [4.1, 5.0, 3.5, 4.2, ...],
#     'gender': ['male', 'female', 'female', 'male', ...]
# })

# Encoding the gender column
label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['gender'])

# Defining the independent variables and adding a constant
X = df[['bty_avg', 'gender_encoded']]
X = sm.add_constant(X)

# Defining the dependent variable
y = df['score_avg']

# Fitting the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Model summary
print(score_bty_gender_fit.summary())

# Interpreting the intercept and slopes
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender: {slope_gender}")

# Scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True, palette='Set1')
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 83 with threat_id: thread_wLMakdUk6PZ5bWTcS5j9yXC5
import pandas as pd
import statsmodels.api as sm

# Load your data into a pandas DataFrame
# df = pd.read_csv("path_to_your_data.csv")

# Prepare the model - assuming 'score' is the dependent variable and 'beauty' + 'gender' are independent variables
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variable 'gender'
y = df['score']
X = sm.add_constant(X)  # adding a constant (intercept)

# Fit the model
model = sm.OLS(y, X).fit()

# Get the model summary to extract R-squared and coefficients
summary = model.summary()

# Extract the R-squared value
r_squared = summary.rsquared
explained_variability_percent = r_squared * 100

# Print out the results
print(f"Percent of variability explained by the model: {explained_variability_percent}%")

# Equation of line for males: assuming 'gender_male' was one of the columns created, and the constant is denoted
coefficients = model.params
intercept = coefficients['const']
beauty_coef = coefficients['beauty']
male_coef = coefficients['gender_male']  # assuming 'gender_male' was created during one-hot encoding

print(f"Equation for male professors: score = {intercept} + {beauty_coef} * beauty + {male_coef}")

# Analyze the relationship using coefficients for the male and female categories
female_coef = -male_coef  # Negative if male coefficient is compared, or directly from coefficients if needed
print(f"Beauty effect on score for Male: {beauty_coef}, for Female: {female_coef}")

# Visualize the data if required
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=df, x='beauty', y='score', hue='gender')
plt.title('Beauty vs Evaluation Score by Gender')
plt.show()
##################################################
#Question 26.1, Round 84 with threat_id: thread_fwTHjNLKcEboK07q8qX6kBkJ
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# Assuming the DataFrame is named 'df' and contains 'score', 'bty_avg', and 'gender' columns.
# df = pd.read_csv('your_data_file.csv')  # Uncomment and modify this line to load your data

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']
X = sm.add_constant(X)  # Add a constant term for the intercept

score_bty_gender_fit = sm.OLS(y, X).fit()  # Fit the model

# Print summary statistics to interpret the coefficients
print(score_bty_gender_fit.summary())

# Interpretation of the coefficients
intercept = score_bty_gender_fit.params[0]
beauty_slope = score_bty_gender_fit.params[1]
gender_slope = score_bty_gender_fit.params[2]  # This depends on how you named your dummy variables for gender

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {beauty_slope}')
print(f'Slope for gender (if male is coded as 1): {gender_slope}')

# Create scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, palette='Set2', dodge=True)
plt.title('Scatterplot of Score vs Beauty Rating by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 84 with threat_id: thread_8zRu0TZeX82hMtZvXfIVFIfk
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load your dataset (modify the path)
data = pd.read_csv('your_data.csv')  # Update with your actual data path

# Filter the dataset for male and female professors
male_professors = data[data['gender'] == 'male']
female_professors = data[data['gender'] == 'female']

# Fit the model
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable to dummy/indicator variables
y = data['evaluation_score']
X = sm.add_constant(X)  # Adds an intercept term to the model

# Fit the model
model = sm.OLS(y, X).fit()

# Percentage of variability explained by the model
r_squared = model.rsquared
print(f"R-squared: {r_squared:.4f}")
percent_variability = r_squared * 100

# Equation of the line for male professors
male_X = male_professors['beauty']
male_y = model.predict(sm.add_constant(pd.get_dummies(male_professors['gender'].replace('male', 1))))

# Coefficients of the model for male
male_coeffs = model.params
male_equation = f"y = {male_coeffs['const']:.4f} + {male_coeffs['beauty']:.4f} * beauty"

# Relationship analysis between beauty and evaluation scores
male_relationship = male_professors[['beauty', 'evaluation_score']].corr().iloc[0, 1]
female_relationship = female_professors[['beauty', 'evaluation_score']].corr().iloc[0, 1]

print(f"Percent of variability explained by the model: {percent_variability:.2f}%")
print(f"Equation of the line for male professors: {male_equation}")
print(f"Correlation between beauty and evaluation score for male professors: {male_relationship:.4f}")
print(f"Correlation between beauty and evaluation score for female professors: {female_relationship:.4f}")
##################################################
#Question 26.1, Round 85 with threat_id: thread_v5VkMwp6lmKR20bbhxy1yBf1
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is in a DataFrame named df and has columns 'score', 'bty_avg', and 'gender'
# df = pd.read_csv('your_data_file.csv')  # Un-comment and modify this line to load your data

# Fit multiple linear regression model
X = df[['bty_avg', 'gender']]
# Convert 'gender' to a numeric variable; e.g., 0 for male, 1 for female (or any encoding you prefer)
X['gender'] = X['gender'].map({'male': 0, 'female': 1})  # Adjust as necessary based on your data
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['score']

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params[0]
slope_bty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create a jittered scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 85 with threat_id: thread_kZS3CYJPENtBZsT2KoWBePpY
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame containing beauty ratings and evaluation scores
# Replace 'score' and 'beauty' with actual column names from your dataset
data = pd.read_csv('your_dataset.csv')

# Model for both genders
X = data['beauty']
y = data['score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(f"R-squared: {model.rsquared:.2f}")  # Percent of variability explained

# Equation of the line for male professors
male_data = data[data['gender'] == 'male']  # Filter for male professors
X_male = male_data['beauty']
y_male = male_data['score']
X_male = sm.add_constant(X_male)
male_model = sm.OLS(y_male, X_male).fit()
print(f"Male professors' line equation: y = {male_model.params[0]:.2f} + {male_model.params[1]:.2f} * beauty")

# Visualization
plt.scatter(data['beauty'], data['score'], c=data['gender'].apply(lambda x: 'blue' if x == 'male' else 'pink'), alpha=0.5)
line_x = np.linspace(data['beauty'].min(), data['beauty'].max(), 100)
line_y_male = male_model.params[0] + male_model.params[1] * line_x
plt.plot(line_x, line_y_male, label='Male line', color='blue')

# Separate models for male and female
female_data = data[data['gender'] == 'female']
X_female = female_data['beauty']
y_female = female_data['score']
X_female = sm.add_constant(X_female)
female_model = sm.OLS(y_female, X_female).fit()
line_y_female = female_model.params[0] + female_model.params[1] * line_x
plt.plot(line_x, line_y_female, label='Female line', color='pink')

plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.title('Beauty vs Evaluation Score by Gender')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 86 with threat_id: thread_ehCyn4GLAWmuN5Llq02ALCM6
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame named 'df' with the necessary columns
# Example: df = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
df['gender'] = df['gender'].astype('category')  # Ensure gender is treated as a categorical variable
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert gender to dummy variables
y = df['score']

X = sm.add_constant(X)  # Adds the intercept
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Adjust if 'Male' is not the higher-coded category

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender (Male): {slope_gender}')

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 86 with threat_id: thread_4yiBb9pkmuoDSj58AB0oRLfo
import statsmodels.api as sm
import pandas as pd

# Assume 'data' is your DataFrame containing 'Score', 'Beauty', and 'Gender'
model = sm.OLS.from_formula('Score ~ Beauty * Gender', data).fit()

# Get R-squared value to determine percent of variability explained
r_squared = model.rsquared
percent_variability_explained = r_squared * 100

# Coefficients for the male professors
male_coef = model.params['Beauty'] + model.params['Beauty:Gender[T.Male]']
intercept_male = model.params['Intercept']
equation_male = f"Score = {intercept_male:.2f} + {male_coef:.2f} * Beauty"

# Print the results
print(f"Percent of variability in score explained: {percent_variability_explained:.2f}%")
print(f"Equation for male professors: {equation_male}")

# Analyzing the relationship differences
relationship_diff = {
    "Male": {"Slope": male_coef, "Intercept": intercept_male},
    "Female": {"Slope": model.params['Beauty'], "Intercept": model.params['Intercept'] + model.params['Gender[T.Female]']}
}

print("Relationship between beauty and evaluation score:")
print(f"Male: Slope = {relationship_diff['Male']['Slope']:.2f}, Intercept = {relationship_diff['Male']['Intercept']:.2f}")
print(f"Female: Slope = {relationship_diff['Female']['Slope']:.2f}, Intercept = {relationship_diff['Female']['Intercept']:.2f}")
##################################################
#Question 26.1, Round 87 with threat_id: thread_uPSWQSS3zSlVQDVAvE5wIX5E
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Assuming you have your data in a CSV file called 'data.csv'
# Replace the filename accordingly.
data = pd.read_csv('data.csv')

# Fit multiple linear regression model
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = data['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'gender_Male' is the dummy variable for Male

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender (Male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, dodge=True, alpha=0.7)
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 87 with threat_id: thread_Px4JyiLBvSFotMFQJr4QZfwK
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with your actual file)
data = pd.read_csv('your_data.csv')  # Ensure you have a CSV file with relevant columns

# Assume the dataset has 'score', 'beauty', and 'gender' columns
# Define the model, e.g. using ordinary least squares (OLS)
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable to dummy variables
y = data['score']

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary which gives us the percent of variability (R-squared)
print(model.summary())

# Extracting the equation for male professors (assuming 'gender_Male' was created)
intercept = model.params['Intercept']
beauty_coef = model.params['beauty']
male_coef = model.params['gender_Male']  # If male is one of the categories

# Equation: score = intercept + beauty_coef * beauty
print(f'Equation for male professors: score = {intercept} + {beauty_coef} * beauty')

# Plotting the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='beauty', y='score', hue='gender')
plt.title('Relationship between Beauty and Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.1, Round 88 with threat_id: thread_6Nyng6bTzi3t0wFeyy91ttRp
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('your_data.csv')

# Assume the dataset has columns: 'average_evaluation_score', 'average_beauty_rating', 'gender'
# Convert gender to a numeric variable if it's not already
data['gender'] = np.where(data['gender'] == 'male', 1, 0)  # male: 1, female: 0

# Define the independent variables (X) and the dependent variable (Y)
X = data[['average_beauty_rating', 'gender']]
Y = data['average_evaluation_score']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(Y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpretation of the coefficients
intercept = score_bty_gender_fit.params[0]
slope_beauty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]

print(f"Intercept: {intercept}")
print(f"Slope for Beauty Rating: {slope_beauty}")
print(f"Slope for Gender: {slope_gender}")

# Create the scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='average_beauty_rating', y='average_evaluation_score', 
                 hue='gender', style='gender', jitter=True)
plt.title('Professor Evaluation Score vs. Beauty Rating by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Evaluation Score')
plt.legend(title='Gender', loc='upper left', labels=['Female', 'Male'])
plt.show()
##################################################
#Question 26.0, Round 88 with threat_id: thread_vlbNVCtZLtlCMntqUDFFiy71
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Make sure to replace 'your_data.csv' with your actual dataset file path
df = pd.read_csv('your_data.csv')  # Change this line to your dataset file

# Fit model
# Assuming the dataset contains a 'beauty', 'score', and 'gender' columns
X = df[['beauty']]  # Predictor variable
X = sm.add_constant(X)  # Adding a constant (intercept) to the model

# Fit model for all data
model = sm.OLS(df['score'], X).fit()

# Get R-squared to determine the percentage of variability explained
percent_variability_explained = model.rsquared * 100

# Fit model for male professors
df_male = df[df['gender'] == 'male']
X_male = df_male[['beauty']]
X_male = sm.add_constant(X_male)
model_male = sm.OLS(df_male['score'], X_male).fit()

# Get the equation of the line for male professors
slope_male = model_male.params['beauty']
intercept_male = model_male.params['const']

# Plotting
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='beauty', y='score', hue='gender', alpha=0.5)
plt.plot(df_male['beauty'], intercept_male + slope_male * df_male['beauty'], color='blue', label='Male Fit', linewidth=2)
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.title('Relationship between Beauty and Evaluation Score')
plt.legend()
plt.show()

# Outputs
print(f'Percentage of Variability Explained (R-squared): {percent_variability_explained:.2f}%')
print(f'Equation of Line (Male Professors): Score = {intercept_male:.2f} + {slope_male:.2f} * Beauty')
##################################################
#Question 26.1, Round 89 with threat_id: thread_DOumYbKc6gFpkNJ2JsmCO5TU
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your data (replace 'your_data.csv' with your actual file)
data = pd.read_csv('your_data.csv')

# Prepare the data
data['gender'] = data['gender'].astype('category')  # Ensure gender is a categorical variable

# Define the independent variables (predictors) and the dependent variable (response)
X = data[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
y = data['score']

# Fit the multiple linear regression model
X = sm.add_constant(X)  # Adding a constant (intercept)
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Male' is one of the genders

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_beauty}')
print(f'Slope for gender (Male): {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=data, jitter=True, dodge=True, alpha=0.7)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 89 with threat_id: thread_AR7gowMIYkAyOtn3dy2S7G5r
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Assume your data is in a CSV file named 'professors_data.csv'
# Make sure to replace 'professors_data.csv' with your actual file path
data = pd.read_csv('professors_data.csv')

# Fit the model for beauty and gender
model = sm.OLS(data['evaluation_score'], sm.add_constant(data[['beauty_score', 'gender']]))
results = model.fit()

# Summary of the model
print(results.summary())

# Percent of variability explained (R-squared)
r_squared = results.rsquared
print(f'Percent of variability explained by the model: {r_squared * 100:.2f}%')

# Equation of the line for male professors only
male_model = sm.OLS(data[data['gender'] == 'male']['evaluation_score'], 
                    sm.add_constant(data[data['gender'] == 'male']['beauty_score']))
male_results = male_model.fit()
slope, intercept = male_results.params

print(f'Equation of the line for male professors: y = {intercept:.2f} + {slope:.2f} * beauty_score')

# Plotting the relationship
plt.figure(figsize=(12, 6))
sns.scatterplot(x=data['beauty_score'], y=data['evaluation_score'], hue=data['gender'], alpha=0.6)
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Beauty vs Evaluation Score by Gender')

# Create line for male professors
x = data[data['gender'] == 'male']['beauty_score']
y = intercept + slope * x
plt.plot(x, y, color='blue', label='Male Professors Fit', linewidth=2)

# Create line for female professors (use the model for female)
female_model = sm.OLS(data[data['gender'] == 'female']['evaluation_score'], 
                      sm.add_constant(data[data['gender'] == 'female']['beauty_score']))
female_results = female_model.fit()
female_slope, female_intercept = female_results.params

# Create line for female professors
x_female = data[data['gender'] == 'female']['beauty_score']
y_female = female_intercept + female_slope * x_female
plt.plot(x_female, y_female, color='red', label='Female Professors Fit', linewidth=2)

plt.legend()
plt.show()
##################################################
#Question 26.1, Round 90 with threat_id: thread_4JYR9JJfEzuiSxfSwopstAXC
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Assume df is your DataFrame that contains 'score', 'bty_avg', and 'gender' columns
# Replace 'your_data.csv' with your actual data file
# df = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score']
X = sm.add_constant(X)  # Add intercept

model = sm.OLS(y, X).fit()
score_bty_gender_fit = model

# Print out the regression results
print(model.summary())

# Interpret coefficients
intercept = model.params['const']
slope_bty = model.params['bty_avg']
slope_gender = model.params['gender_Male']  # Adjust based on how gender was converted to dummies

print(f"Intercept: {intercept:.3f}")
print(f"Slope for Beauty Rating: {slope_bty:.3f}")
print(f"Slope for Gender (Male): {slope_gender:.3f}")

# Creating a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True, palette='Set2')
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 90 with threat_id: thread_0sQloP98bSTEBxCePGtgtaOD
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('your_dataset.csv')  # Replace with your data file

# Fit the model: Assume 'evaluation_score' is the dependent variable and 'beauty' and 'gender' are independent
model = sm.OLS.from_formula('evaluation_score ~ beauty * gender', data).fit()

# Summary of the model
print(model.summary())

# Get R-squared to determine the percent of variability
r_squared = model.rsquared * 100
print(f"Percentage of variability explained: {r_squared:.2f}%")

# Equation for male professors only
male_model = sm.OLS.from_formula('evaluation_score ~ beauty', data[data['gender'] == 'male']).fit()
print(f"Equation for males: Evaluation Score = {male_model.params['Intercept']:.2f} + {male_model.params['beauty']:.2f} * Beauty")

# Visualizing the relationship
sns.lmplot(x='beauty', y='evaluation_score', hue='gender', data=data)
plt.title('Relationship between Beauty and Evaluation Score')
plt.xlabel('Beauty')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 91 with threat_id: thread_S7Iw4QUiml9FvN90Row0QHj2
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame `df` with columns 'score_avg', 'bty_avg', and 'gender'

# Step 1: Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score_avg']

X = sm.add_constant(X)  # Add intercept

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Step 2: Print the model summary for interpretation
print(score_bty_gender_fit.summary())

# Intercept and coefficients interpretation
intercept = score_bty_gender_fit.params[0]
beauty_coefficient = score_bty_gender_fit.params[1]  # Coefficient for 'bty_avg'
gender_coefficient = score_bty_gender_fit.params[2]  # Coefficient for 'gender' (dummy variable)

print(f"Intercept: {intercept}")
print(f"Beauty Rating Coefficient: {beauty_coefficient}")
print(f"Gender Coefficient: {gender_coefficient}")

# Step 3: Create jittered scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 91 with threat_id: thread_rJ4sKbGAvG1uAzJVNPHXulDe
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_data.csv' with your actual data file)
data = pd.read_csv('your_data.csv')

# Fit regression model
X = data[['beauty', 'gender']]  # Assuming 'beauty' and 'gender' are columns
y = data['score']
X = pd.get_dummies(X, columns=['gender'], drop_first=True)  # One-hot encoding for gender

# Add a constant for intercept
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

# Extract coefficients
male_coef = model.params['gender_Male']  # Assuming male is encoded as 'gender_Male'
intercept = model.params['const']

# Equation for male professors
equation_male = f"Score = {intercept} + {male_coef} * Beauty"

# Visualization
sns.lmplot(data=data, x='beauty', y='score', hue='gender')
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()

# Print the equation
print("Equation for male professors:", equation_male)
##################################################
#Question 26.1, Round 92 with threat_id: thread_lrMY8a1Vp34gJ93ZK3rf4Ue3
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into a DataFrame (assuming the data is in a CSV file)
# df = pd.read_csv('your_data.csv')

# For example purposes, let's assume df is already defined as:
# df = pd.DataFrame({
#     'score': [4.5, 3.0, 4.0, 4.7, 2.5],
#     'bty_avg': [7, 6, 8, 9, 5],
#     'gender': ['F', 'M', 'F', 'M', 'F']
# })

# Convert gender to numerical
df['gender'] = df['gender'].map({'F': 0, 'M': 1})  # F=0, M=1

# Define the independent variables and dependent variable
X = df[['bty_avg', 'gender']]
y = df['score']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(score_bty_gender_fit.summary())

# Interpretation of the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_bty_avg = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f"Intercept (score when bty_avg=0 and gender=0): {intercept}")
print(f"Slope of beauty rating: {slope_bty_avg} (Change in score for 1 unit increase in beauty rating)")
print(f"Slope of gender (1 if male, 0 if female): {slope_gender} (Change in score for being male vs female)")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()
##################################################
#Question 26.0, Round 92 with threat_id: thread_eElm0O3Evq9FM0FWcjHmy0kM
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('your_data.csv')  # Replace with your actual data file

# Fit the model score by beauty and gender
model = sm.OLS(df['evaluation_score'], sm.add_constant(df[['beauty_score', 'gender']]))
results = model.fit()

# Percent of variability explained by the model (R-squared)
r_squared = results.rsquared
print(f"Percent of variability explained by the model: {r_squared * 100:.2f}%")

# Equation of the line for male professors
male_professors = df[df['gender'] == 'male']
male_model = sm.OLS(male_professors['evaluation_score'], sm.add_constant(male_professors[['beauty_score']])).fit()
male_intercept, male_slope = male_model.params

print(f"Equation for Male Professors: evaluation_score = {male_intercept:.2f} + {male_slope:.2f} * beauty_score")

# Relationship between beauty and evaluation score for males and females
sns.lmplot(data=df, x='beauty_score', y='evaluation_score', hue='gender', markers=["o", "s"], palette="Set1")
plt.title('Evaluations by Beauty Score and Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 93 with threat_id: thread_J5hIZWwFdtthwQeFpOXou3fn
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data into a Pandas DataFrame (replace 'your_data.csv' with your actual data file)
data = pd.read_csv('your_data.csv')

# Fit the multiple linear regression model
X = data[['bty_avg', 'gender']]  # predictors
X = pd.get_dummies(X, drop_first=True)  # convert categorical variable 'gender' into dummy/indicator variables
y = data['score_avg']  # response variable

X = sm.add_constant(X)  # adds a constant term to the predictors

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Intercept and slopes interpretation
intercept = score_bty_gender_fit.params[0]
slope_beauty = score_bty_gender_fit.params[1]
slope_gender = score_bty_gender_fit.params[2]  # assuming gender was the second variable

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_beauty}')
print(f'Slope for gender: {slope_gender}')

# Create a scatter plot of score by bty_avg colored by gender
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 93 with threat_id: thread_Rvp3a9A5Ayi9m6Q0SDHG6BQH
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data Loading (replace with your actual dataset)
# df = pd.read_csv('your_data.csv')  # Load a dataset that contains 'beauty', 'score', and 'gender'.

# Example DataFrame Creation (For demonstration purposes)
data = {
    'beauty': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'score': [1.5, 2.0, 2.5, 3.6, 5.6, 6.5, 6.8, 8.0, 8.5, 9.0],
    'gender': ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male']
}
df = pd.DataFrame(data)

# Fit the model
X = df[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert 'gender' into dummy variables
y = df['score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()  # Fit the regression model

# Summary of the model
print(model.summary())

# Calculate explained variance (R-squared)
explained_variance = model.rsquared
print(f'Percent of variance explained by the model: {explained_variance * 100:.2f}%')

# Equation for Male professors (assuming 'gender_male' is the dummy variable created)
intercept = model.params['const']
b_beauty = model.params['beauty']
b_gender_male = model.params['gender_male']

# Equation: score = intercept + b_beauty * beauty + b_gender_male * (1 if male else 0)
male_equation = f'Score = {intercept:.2f} + {b_beauty:.2f} * beauty + {b_gender_male:.2f}'
print(f'Equation for Male Professors: {male_equation}')

# Visualizing the relationship
sns.lmplot(x='beauty', y='score', hue='gender', data=df)
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.show()
##################################################
#Question 26.1, Round 94 with threat_id: thread_cCX9Ife3AQlTNN5w3abcwzOu
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataframe
# your_dataframe = pd.read_csv('your_datafile.csv')  # Example of loading data

# Assuming the DataFrame has columns: 'score', 'bty_avg', 'gender'
# Rename columns for clarity
df = your_dataframe[['score', 'bty_avg', 'gender']]

# Map gender to numerical values (if it's not already numeric)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})  # or similar mapping

# Define features and target
X = df[['bty_avg', 'gender']]
y = df['score']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the model coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender']

print(f'Intercept: {intercept}')
print(f'Slope for beauty rating: {slope_bty}')
print(f'Slope for gender: {slope_gender}')

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()
##################################################
#Question 26.0, Round 94 with threat_id: thread_2bT58SEWRQgNPAfklhLYx3j7
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample data to exemplify beauty scores and evaluation scores for male and female professors
data = {
    'beauty_score': [1, 2, 3, 4, 5, 6],
    'evaluation_score_male': [2, 3, 5, 3, 6, 7],
    'evaluation_score_female': [3, 4, 2, 5, 6, 8]
}
df = pd.DataFrame(data)

# Fitting model for male professors
X_male = sm.add_constant(df['beauty_score'])
model_male = sm.OLS(df['evaluation_score_male'], X_male).fit()

# Fitting model for female professors
X_female = sm.add_constant(df['beauty_score'])
model_female = sm.OLS(df['evaluation_score_female'], X_female).fit()

# Output the summaries
print("Male Model Summary:")
print(model_male.summary())
print("\nFemale Model Summary:")
print(model_female.summary())

# Plotting
plt.scatter(df['beauty_score'], df['evaluation_score_male'], color='blue', label='Male Evaluation')
plt.scatter(df['beauty_score'], df['evaluation_score_female'], color='red', label='Female Evaluation')
plt.plot(df['beauty_score'], model_male.predict(X_male), color='blue', linestyle='--', label='Male Fit')
plt.plot(df['beauty_score'], model_female.predict(X_female), color='red', linestyle='--', label='Female Fit')
plt.xlabel('Beauty Score')
plt.ylabel('Evaluation Score')
plt.title('Beauty vs Evaluation Score')
plt.legend()
plt.show()
##################################################
#Question 26.1, Round 95 with threat_id: thread_8XhpLDBIFJwecqSjMCyKBQLG
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your data
# Replace 'your_data.csv' with your actual data file
# df = pd.read_csv('your_data.csv')

# In this example, we will generate a DataFrame for demonstration
# You should replace this with your actual data loading
data = {
    'score_avg': [4.5, 3.0, 4.0, 3.5, 4.7, 4.2, 3.8],
    'bty_avg': [4.0, 3.5, 4.2, 3.8, 4.5, 4.1, 3.7],
    'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
}
df = pd.DataFrame(data)

# Convert gender to a numeric variable for regression
df['gender_numeric'] = df['gender'].map({'Female': 1, 'Male': 0})

# Fit the multiple linear regression model
X = df[['bty_avg', 'gender_numeric']]
X = sm.add_constant(X) # Adds a constant term to the predictor
y = df['score_avg']
model_score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(model_score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = model_score_bty_gender_fit.params['const']
slope_bty = model_score_bty_gender_fit.params['bty_avg']
slope_gender = model_score_bty_gender_fit.params['gender_numeric']

print(f'Intercept: {intercept}')
print(f'Slope for Beauty Rating: {slope_bty}')
print(f'Slope for Gender: {slope_gender}')

# Create a scatter plot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', data=df, jitter=True, hue='gender', dodge=True)

plt.title('Average Professor Evaluation Score by Beauty Rating and Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 95 with threat_id: thread_CzXFyX4xlchaXGoYi5IroIC3
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Assuming df is your DataFrame with 'evaluation_score', 'beauty', and 'gender' columns.
# Load your data
# df = pd.read_csv('your_data.csv')

# Fit the model
X = df[['beauty', 'gender_male']]  # Use gender as a binary variable (0/1)
y = df['evaluation_score']

X = sm.add_constant(X)  # Adds the constant term to the predictor
model = sm.OLS(y, X).fit()

# Variability explained
r_squared = model.rsquared
percent_variability = r_squared * 100

# Equation of the line for male professors
coef = model.params
male_equation = f"y = {coef['const']} + {coef['beauty']} * beauty + {coef['gender_male']} * 1"

# Analyzing the relationship for male and female professors
relationship_summary = model.summary()

print(f"Percent of variability explained: {percent_variability:.2f}%")
print(f"Equation for male professors: {male_equation}")
print(relationship_summary)  # This will give you detailed statistics about the model
##################################################
#Question 26.1, Round 96 with threat_id: thread_iuse0R0MLQ77uJ53RG0HtSg4
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Replace 'your_data_file.csv' with the path to your actual data file
df = pd.read_csv('your_data_file.csv')

# Prepare the data
# Assuming the DataFrame has columns 'score', 'bty_avg', and 'gender'
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' into dummy/indicator variables
y = df['score']

# Fit the multiple linear regression model
X = sm.add_constant(X)  # Adding a constant to the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print model summary
print(score_bty_gender_fit.summary())

# Interpret the intercept and slopes
intercept = score_bty_gender_fit.params['const']
slope_beauty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_Male']  # Assuming 'Male' is one of the levels of gender

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_beauty}")
print(f"Slope for gender (Male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score', hue='gender', data=df, jitter=True, palette='Set2')
plt.title('Scatterplot of Average Professor Evaluation Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 96 with threat_id: thread_nYYcdFmfdZQhqCABkbTBI3PR
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# data = pd.read_csv('your_dataset.csv')  # Uncomment and replace with your dataset

# Example data
data = pd.DataFrame({
    'score': [3, 5, 6, 7, 2, 4, 5, 3, 6, 4],
    'beauty': [7, 8, 6, 9, 4, 5, 6, 7, 8, 9],
    'gender': ['male', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male']
})

# Fit the model
X = data[['beauty', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
y = data['score']

model = sm.OLS(y, sm.add_constant(X)).fit()

# Output the summary
print(model.summary())

# Percentage of variability explained
r_squared = model.rsquared * 100

# Get the equation for male professors (gender_male = 1)
male_coefs = model.params['const'] + model.params['beauty'] * data[data['gender'] == 'male']['beauty'].mean()
print(f"Equation of the line for male professors: y = {male_coefs:.2f}")

# Plotting the results
sns.lmplot(data=data, x='beauty', y='score', hue='gender', markers=["o", "x"])
plt.title('Beauty vs Evaluation Score by Gender')
plt.xlabel('Beauty')
plt.ylabel('Score')
plt.show()

# Final output
output = {
    "percent_explained": r_squared,
    "equation_male": f"y = {male_coefs:.2f}",
    "model_summary": model.summary().as_text()
}

print(output)
##################################################
#Question 26.1, Round 97 with threat_id: thread_pzGNueybQzy8YEcbikuRMaJA
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into a DataFrame
# df = pd.read_csv('your_data_file.csv') # Un-comment and adjust this line to load your dataset

# Assuming the dataframe 'df' has columns 'score_avg', 'bty_avg', and 'gender'

# Prepare the data
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable 'gender' to dummy variables
y = df['score_avg']

# Fit the multiple linear regression model
X = sm.add_constant(X)  # Adds a constant term to the predictor
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the model summary
print(score_bty_gender_fit.summary())

# Interpretation of coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_1']  # Adjust based on your dummy variable names

print(f"Intercept (constant): {intercept:.3f}")
print(f"Slope for beauty rating: {slope_bty:.3f}")
print(f"Slope for gender (if gender is male in this case): {slope_gender:.3f}")

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, palette='Set2')
plt.title('Scatterplot of Average Score by Average Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 97 with threat_id: thread_fKxrMOsGj1tyiMyoK4E6nr6g
import pandas as pd
import statsmodels.api as sm

# Sample DataFrame creation
# df = pd.read_csv('your_data.csv')  # Load your data here

# For example, let's assume you have the following columns: 'score', 'beauty', 'gender'
# df['gender'] = df['gender'].astype('category')

# Fit the model
model = sm.OLS.from_formula('score ~ beauty + gender', data=df).fit()

# Get the percent of variability explained by the model
rsq = model.rsquared
print(f"Percent of variability explained by the model: {rsq * 100:.2f}%")

# Equation of the line for male professors
# Extract coefficients for male gender
male_coef = model.params['Intercept'] + model.params['gender[T.male]']  # Base Intercept for males
beauty_coef = model.params['beauty']
print(f"Equation of the line for male professors: score = {male_coef} + {beauty_coef} * beauty")

# Analyzing the relationship between beauty and evaluation score for male and female professors
print(model.summary())
##################################################
#Question 26.1, Round 98 with threat_id: thread_NuKcuMa9tOqVSG3c5jQU7M4x
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (replace 'data.csv' with your actual data file)
# df = pd.read_csv('data.csv')

# Assuming your dataframe has columns named 'score_avg', 'bty_avg', and 'gender'
# Fit the multiple linear regression model
X = df[['bty_avg', 'gender']]
X = pd.get_dummies(X, drop_first=True)  # Converting categorical variable 'gender' to dummy variables
y = df['score_avg']

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the model
score_bty_gender_fit = sm.OLS(y, X).fit()

# Print the summary of the model
print(score_bty_gender_fit.summary())

# Interpreting coefficients
intercept = score_bty_gender_fit.params['const']
slope_bty = score_bty_gender_fit.params['bty_avg']
slope_gender = score_bty_gender_fit.params['gender_male']  # Assuming 'gender_male' is the dummy for male

print(f"Intercept: {intercept}")
print(f"Slope for beauty rating: {slope_bty}")
print(f"Slope for gender (male): {slope_gender}")

# Create a scatterplot with jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=df, jitter=True, dodge=True)
plt.title('Scatterplot of Average Professor Evaluation Score by Beauty Rating')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
##################################################
#Question 26.0, Round 99 with threat_id: thread_nt3fhXMuoeDyqmM3a6VZ859k
# Required Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data Creation (Assuming your data is loaded in a DataFrame named 'data')
# Replace this with your actual data loading step
# data = pd.read_csv('your_file.csv')

# Sample data for illustration
data = pd.DataFrame({
    'score_avg': np.random.uniform(1, 5, 100),  # Average professor evaluation score
    'bty_avg': np.random.uniform(1, 5, 100),    # Average beauty rating
    'gender': np.random.choice(['Male', 'Female'], 100)  # Gender
})

# Convert gender to numerical values for regression
data['gender_num'] = data['gender'].map({'Male': 0, 'Female': 1})

# Define the dependent and independent variables
X = data[['bty_avg', 'gender_num']]
y = data['score_avg']

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()
score_bty_gender_fit = model

# Print the summary of the regression
print(score_bty_gender_fit.summary())

# Interpret the coefficients
intercept = model.params['const']
slope_bty = model.params['bty_avg']
slope_gender = model.params['gender_num']

print(f'Intercept: {intercept:.2f}')
print(f'Slope for beauty rating: {slope_bty:.2f}')
print(f'Slope for gender (Female vs Male): {slope_gender:.2f}')

# Jittered scatterplot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bty_avg', y='score_avg', hue='gender', data=data, dodge=True, jitter=True, alpha=0.7)
plt.title('Professor Evaluation Score by Beauty Rating Colored by Gender')
plt.xlabel('Average Beauty Rating')
plt.ylabel('Average Professor Evaluation Score')
plt.legend(title='Gender')
plt.show()
