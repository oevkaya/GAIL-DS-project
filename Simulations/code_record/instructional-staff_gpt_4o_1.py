##################################################
#Question 14, Round 0 with threat_id: thread_6lKBk1bSekSSn6ehOCbHCqGM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb')

# Transform the data to long format
long_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert year to integer
long_data['Year'] = long_data['Year'].astype(int)

# Create the dot plot
plt.figure(figsize=(12, 6))
sns.scatterplot(
data=long_data,
x='Year',
y='Percentage',
hue='faculty_type',
palette='tab10',
s=100
)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975 - 2011)')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Faculty')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 1 with threat_id: thread_dNKI1dlMB6XKEp16BQRbxG8h
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb')

# Reshape the dataframe for plotting
df_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_long, x='year', y='percentage', hue='faculty_type', s=100, palette='tab10', marker='o')

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 4 with threat_id: thread_xbvKhBjextm8vCnoNyD5UFVw
import matplotlib.pyplot as plt

# Prepare the data
years = data.columns[1:]  # Years are the column names except the first one
faculty_types = data['faculty_type']  # Faculty type is in the first column

# Start plotting
plt.figure(figsize=(10, 6))

# Iterate over each faculty type and plot the data
for index, row in data.iterrows():
plt.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Add titles and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 5 with threat_id: thread_jsWDWY2pmjCvmviEvutddmUE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

# Set the figure size
plt.figure(figsize=(14, 8))

# Melt the dataframe to have a long format suitable for seaborn
long_data = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' column to integers for better plotting
long_data['Year'] = long_data['Year'].astype(int)

# Create a dot plot using seaborn's pointplot
sns.pointplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', dodge=True, markers='o', linestyles='')

# Set title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)

# Display the legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 6 with threat_id: thread_IxKlqWDUIOBaaYADw49eOp5Z
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded file and examine its contents
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
df = pd.read_csv(file_path)

# Set up the figure size and style
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Melt the dataframe to long format
df_long = df.melt(id_vars="faculty_type", var_name="year", value_name="percentage")

# Convert year to integer for sorting
df_long['year'] = df_long['year'].astype(int)

# Create a dot plot
sns.scatterplot(data=df_long, x="year", y="percentage", hue="faculty_type", style="faculty_type", s=100, palette="tab10")

# Customize the plot
plt.title("Instructional Staff Employment Trends (1975-2011)")
plt.xlabel("Year")
plt.ylabel("Percentage of Total Instructional Staff")
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(df_long['year'].unique())
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 7 with threat_id: thread_3nEKNCXiofKk8vJYQDS8JKA9
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb')

# Setting up the plot
plt.figure(figsize=(12, 8))
years = data.columns[1:]  # Exclude 'faculty_type' from the year list

# Plot each faculty type as a separate line
for index, row in data.iterrows():
plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Customizing the plot
plt.title('Instructional Staff Employment Trends (1975 - 2011)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 9 with threat_id: thread_6bBsulUfrNVVsEB3yummL22e
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
df = pd.read_csv(file_path)

# Transform the data for plotting
df_melted = df.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Change the 'Year' column to numeric for plotting
df_melted['Year'] = pd.to_numeric(df_melted['Year'])

# Define a color palette
palette = sns.color_palette("husl", len(df['faculty_type']))

# Create the dot plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_melted, x='Year', y='Percentage', hue='faculty_type', palette=palette, s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Instructional Staff', fontsize=12)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 11 with threat_id: thread_vALEMCJsmswPhICFviEU3HOW
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the uploaded file
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Melt the data for easier plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')
data_melted['year'] = data_melted['year'].astype(int)

# Generate the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Add title and labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.grid()
plt.show()
##################################################
#Question 14, Round 12 with threat_id: thread_ZBYim83vwG3PqoNbbbicPGig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
staff_data = pd.read_csv(file_path, encoding='utf-8-sig')

# Prepare the data for plotting by melting it
melted_data = staff_data.melt(id_vars=['faculty_type'], var_name='Year', value_name='Employment_Rate')

# Create a dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=melted_data, x='Year', y='Employment_Rate', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Rate (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 13 with threat_id: thread_2MknNBLntJiJR7xhMp0KGFaw
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Prepare the data
years = data.columns[1:].astype(int)  # Extract years from the columns
faculty_types = data['faculty_type']  # Extract faculty types

# Plotting
plt.figure(figsize=(10, 6))
for index, row in data.iterrows():
plt.plot(years, row[1:], marker='o', label=row['faculty_type'])

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage Employment')
plt.xticks(years, rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(visible=True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 14 with threat_id: thread_0oSDLpRac7Atn7fL7ETcIRXS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Set the style of the plot
sns.set(style="whitegrid")

# Melt the dataframe to have 'Year' as a variable for plotting
data_melted = pd.melt(data, id_vars="faculty_type", var_name="Year", value_name="Percentage")
data_melted['Year'] = data_melted['Year'].astype(int)

# Create the dot plot
plt.figure(figsize=(12, 6))
dotplot = sns.scatterplot(
data=data_melted,
x="Year",
y="faculty_type",
size="Percentage",
hue="Percentage",
palette="viridis",
sizes=(20, 200),
legend=None,
alpha=0.7
)

# Customize the plot details
dotplot.set_title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
dotplot.set_xlabel('Year', fontsize=12)
dotplot.set_ylabel('Faculty Type', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 15 with threat_id: thread_BFclvebr2sB2qgv7gl4EXVPo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Prepare the data for plotting
years = data.columns[1:].astype(int)  # Extract years from columns, convert to integers
faculty_types = data['faculty_type']  # Faculty types
num_faculty_types = len(faculty_types)

# Initialize the plot
plt.figure(figsize=(12, 8))

# Plot each faculty type's trend with dots
for i, faculty_type in enumerate(faculty_types):
employment_trend = data.iloc[i, 1:]  # Employment trend data for the current faculty type
plt.scatter(years, employment_trend, label=faculty_type)
plt.plot(years, employment_trend, linestyle='--', alpha=0.5)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(years, rotation=45)
plt.yticks(np.arange(0, 50, 5))
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 17 with threat_id: thread_l4I6CPLIQn9KY4ilf9RWVQzM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Melt the data to make it suitable for seaborn plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Employment_Percentage')

# Convert "Year" to integer for proper sorting
melted_data['Year'] = melted_data['Year'].astype(int)

# Create a dot plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Plot using seaborn
dot_plot = sns.scatterplot(
data=melted_data,
x='Year',
y='Employment_Percentage',
hue='faculty_type',
palette='tab10',
s=100  # size of the dots
)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage', fontsize=14)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(melted_data['Year'].unique(), rotation=45)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 18 with threat_id: thread_IX9b2fB4RpTQZhMWN7dVsUmE
import pandas as pd

# Load the file
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Display the first few rows of the data to inspect its structure
data.head()


import matplotlib.pyplot as plt

# Transpose the data to have years as index for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert year back to integer from string caused by melt
data_melted['year'] = data_melted['year'].astype(int)

# Plotting
plt.figure(figsize=(12, 8))
for faculty_type in data['faculty_type']:
subset = data_melted[data_melted['faculty_type'] == faculty_type]
plt.plot(subset['year'], subset['percentage'], marker='o', label=faculty_type)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 21 with threat_id: thread_Czdc1pEhPq4LiIxk8sPAjqDg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data file
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Melt the data to have a long-form DataFrame suitable for Seaborn
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the year to an integer for proper sorting
data_melted['Year'] = data_melted['Year'].astype(int)

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.stripplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', jitter=0.2, dodge=True, marker='o', alpha=0.7)

# Enhance the plot with titles and legends
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage', fontsize=12)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 23 with threat_id: thread_NtOi1QJWuvGZwOpupuZ5IuwJ
import pandas as pd
import matplotlib.pyplot as plt

# Load and inspect the first few rows of the uploaded file
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Preparing for the dot plot
years = data.columns[1:]  # Extract years from the columns
faculty_types = data['faculty_type']

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

for index, row in data.iterrows():
ax.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Adding titles, labels, and legend
ax.set_title('Trends in Instructional Staff Employment (1975-2011)')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Total Instructional Staff')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 24 with threat_id: thread_1wlrkzYwVLapV7nIkhwENX8I
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Set the plot size
plt.figure(figsize=(12, 8))

# Iterate over each faculty type and plot its employment trend
for index, row in data.iterrows():
years = data.columns[1:].astype(int)  # Get the years (converted to int)
employment_trend = row[1:].values     # Get the employment trend data
plt.plot(years, employment_trend, 'o-', label=row['faculty_type'], markersize=8)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage', fontsize=12)
plt.xticks(years, rotation=45)
plt.yticks(np.arange(0, 50, 5))
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 25 with threat_id: thread_kNNaBiVAqiYVtiiVjnnrfUsb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded file into a DataFrame
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Transpose the data for plotting
data_long = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Convert the 'Year' column to integers for proper plotting
data_long['Year'] = data_long['Year'].astype(int)

# Set the style of the plot
sns.set(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100, palette='tab10')

# Add labels and title
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage', fontsize=14)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 26 with threat_id: thread_757HeBdJJFhmR0Z7VueNwem2
import matplotlib.pyplot as plt

# Transpose the data to have years on the x-axis
data_long = data.set_index('faculty_type').transpose()

# Plot the dot plot
plt.figure(figsize=(12, 8))
for faculty_type in data_long.columns:
plt.plot(data_long.index, data_long[faculty_type], 'o-', label=faculty_type)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 27 with threat_id: thread_llI06D6dfEJKcmZJC4oaCryk
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-WBXRxqsnGtjL1x3ZQf5Peb'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot each faculty type as a series of dots
for index, row in data.iterrows():
years = data.columns[1:]  # Years are the columns excluding 'faculty_type'
values = row[1:]          # Values are the remaining columns in each row

# Plot a line of dots
plt.plot(years, values, marker='o', linestyle='', label=row['faculty_type'])

# Add plot labels and title
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 30 with threat_id: thread_anf2MQ9W5h24B9RVTWGl3luo
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/path/to/your/file.csv'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(12, 8))

# Define the years
years = data.columns[1:]

# Plot each faculty type
for index, row in data.iterrows():
    plt.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 31 with threat_id: thread_GJDGIzr7r8rxl1ebo1idLglL
import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
data = pd.read_csv('your_file.csv')

# Transform the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the plot
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10.colors

# Plot each faculty type as a separate series
for index, (label, group_data) in enumerate(data_melted.groupby('faculty_type')):
    plt.plot(group_data['Year'], group_data['Percentage'], 'o-', label=label, color=colors[index])

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 32 with threat_id: thread_K78y8ebUdztEBmEvDP3rZCl6
import matplotlib.pyplot as plt
import seaborn as sns

# Ensuring trends data is in long format for easier plotting
long_data = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Initializing the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Creating the dot plot
sns.scatterplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', s=100, palette='tab10')

# Add titles and labels
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 33 with threat_id: thread_sAjSAS7htICOCWlxUpjKlBh5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/your/file/path.csv' # Update this path if needed
data = pd.read_csv(file_path)

# Melt the data to make it suitable for dot plot
df_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the dot plot
plt.figure(figsize=(12, 8))

# Plot each faculty type
for index, row in data.iterrows():
    plt.plot(data.columns[1:], row[1:], marker='o', label=row['faculty_type'])

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 34 with threat_id: thread_0AllwklXHMEYQuT7ROJl91rI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('your_file_path.csv')  # Ensure to use your correct file path

# Prepare the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the dot plot using seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Enhance the plot with titles and labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 35 with threat_id: thread_ZThbDnnhKKvOTEaermJTdo5a
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot the data as a dot plot
for index, row in data.iterrows():
    plt.plot(row.index[1:], row.values[1:], 'o-', label=row['faculty_type'])

# Customize the plot
plt.title('Trends in Instructional Staff Employment')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 36 with threat_id: thread_9YZRJK6pEhYQQ9lSDJLljCg0
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/path/to/your/file.csv'  # Update with the actual file path
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' column to integer for plotting
data_long['Year'] = data_long['Year'].astype(int)

# Plotting
plt.figure(figsize=(10, 6))
for faculty_type in data['faculty_type']:
    plt.plot(
        data_long[data_long['faculty_type'] == faculty_type]['Year'], 
        data_long[data_long['faculty_type'] == faculty_type]['Percentage'], 
        marker='o', 
        label=faculty_type)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Faculty Type')
plt.xticks(data_long['Year'].unique())  # Ensure all years are shown on the x-axis
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 37 with threat_id: thread_oTiOa3MoaCMiVMoMXpCZ50pf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to transform years into a variable
data_melted = data.melt(id_vars="faculty_type", var_name="year", value_name="percentage")

# Convert the 'year' column to numeric for sorting purposes
data_melted['year'] = pd.to_numeric(data_melted['year'])

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
##################################################
#Question 14, Round 38 with threat_id: thread_2mBdHXZuj7Kv4Xkzfuex5fvL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transform the data to a long format for plotting with seaborn
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the year to a numeric type
data_long['Year'] = data_long['Year'].astype(int)

# Create a dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', palette='tab10', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 39 with threat_id: thread_mMTtrcYgShjIOXOwy829jcDS
import pandas as pd
import matplotlib.pyplot as plt

# Data from the uploaded file
data = pd.DataFrame({
    "faculty_type": ["Full-Time Tenured Faculty", "Full-Time Tenure-Track Faculty",
                     "Full-Time Non-Tenure-Track Faculty", "Part-Time Faculty", 
                     "Graduate Student Employees"],
    1975: [29.0, 16.1, 10.3, 24.0, 20.5],
    1989: [27.6, 11.4, 14.1, 30.4, 16.5],
    1993: [25.0, 10.2, 13.6, 33.1, 18.1],
    1995: [24.8, 9.6, 13.6, 33.2, 18.8],
    1999: [21.8, 8.9, 15.2, 35.5, 18.7],
    2001: [20.3, 9.2, 15.5, 36.0, 19.0],
    2003: [19.3, 8.8, 15.0, 37.0, 20.0],
    2005: [17.8, 8.2, 14.8, 39.3, 19.9],
    2007: [17.2, 8.0, 14.9, 40.5, 19.5],
    2009: [16.8, 7.6, 15.1, 41.1, 19.4],
    2011: [16.7, 7.4, 15.4, 41.3, 19.3]
})

# Transposing for easier plotting
data = data.set_index('faculty_type').transpose()

# Plotting the dot plot
plt.figure(figsize=(12, 8))

for column in data.columns:
    plt.plot(data.index, data[column], marker='o', label=column)

plt.title("Instructional Staff Employment Trends Over the Years")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.xticks(data.index)  # Ensuring all years are shown
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 40 with threat_id: thread_4p4GO6CIXR1ly8D7YJ1pjukZ
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading (replace this line with loading your specific data, as we've done above)
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format for easier plotting with seaborn
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert year column to integer
data_melted['year'] = data_melted['year'].astype(int)

# Set the style of the visualization
sns.set(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Add titles and labels
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 41 with threat_id: thread_nm4VHMAjMIBIzfm6T5un9NEg
import matplotlib.pyplot as plt

# Set the style and figure size for better aesthetics
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 8))

# Transpose the dataframe for easier plotting
data_long = data.set_index('faculty_type').T

# Plotting each faculty type
markers = ['o', 's', 'D', '^', 'v']  # Different markers for each line
for i, faculty in enumerate(data_long.columns):
    plt.plot(data_long.index, data_long[faculty], marker=markers[i % len(markers)], label=faculty)

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Percentage or Count')
plt.title('Instructional Staff Employment Trends')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Displaying the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 42 with threat_id: thread_9HPm1gAMgHT20OG3kL3xpFb6
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to integer for sorting
data_melted['Year'] = data_melted['Year'].astype(int)

# Plotting
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create the plot
sns.scatterplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Title and labels
plt.title('Instructional Staff Employment Trends', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage (%)', fontsize=12)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 43 with threat_id: thread_7rwaj4xc6R9IN4bLp9ZAGW56
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to long format for easy plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the Year column to numeric
data_melted['Year'] = pd.to_numeric(data_melted['Year'])

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 44 with threat_id: thread_CtjSMngAhyTb6TpB0tCHi37l
import matplotlib.pyplot as plt

# Prepare data
faculty_types = data['faculty_type']
years = data.columns[1:]
data_values = data.iloc[:, 1:]

# Plot
plt.figure(figsize=(10, 6))

for index, row in enumerate(data_values.values):
    plt.plot(years, row, 'o-', label=faculty_types[index])

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 45 with threat_id: thread_KS8orT1DnrEmuMTeZVKbyqNX
import matplotlib.pyplot as plt

# Prepare the data for plotting
years = data.columns[1:]
faculty_types = data['faculty_type']

# Plotting the dot plot
plt.figure(figsize=(12, 8))

# Plot each faculty type as a series of dots over the years
for index, row in data.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Adding labels and title
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(loc='best', fontsize='small')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 46 with threat_id: thread_0v7faAQ1Ojckf4xRCBMXCsCQ
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('your_file.csv')  # Ensure to replace 'your_file.csv' with the correct file path

# Set the style of the visualization
sns.set(style="whitegrid")

# Melt the DataFrame to long format
long_data = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Create the dot plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=long_data, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Add titles and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')

# Improve legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 47 with threat_id: thread_BkFic2dxgFYzDFdaTGl87Opj
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for plotting
data_transposed = data.set_index('faculty_type').T
data_transposed.index = data_transposed.index.astype(int)

# Create the dot plot
plt.figure(figsize=(10, 6))

for faculty_type in data_transposed.columns:
    plt.plot(data_transposed.index, data_transposed[faculty_type], 'o-', label=faculty_type)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(data_transposed.index, rotation=45)
plt.yticks(range(0, 51, 5))
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 48 with threat_id: thread_bfHYhg1w4PhQ7af6WKaQL96F
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot each faculty type
for i, row in data.iterrows():
    years = data.columns[1:]  # All years
    values = row[1:]          # Corresponding values for each year
    plt.plot(years, values, '-o', label=row['faculty_type'])

# Customize plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 49 with threat_id: thread_5XXeywnRCzp8J8O2AUsOeGz3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'path_to_your_file.csv'
data = pd.read_csv(file_path)

# Melt the data to long format
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert year to int for sorting
data_long['Year'] = data_long['Year'].astype(int)

# Initialize the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a dot plot
sns.scatterplot(data=data_long, x='Year', y='faculty_type', size='Percentage', hue='faculty_type', sizes=(20, 200), legend=False)

# Add title and labels
plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Faculty Type')
plt.grid(True)

# Show plot
plt.show()
##################################################
#Question 14, Round 50 with threat_id: thread_LE3W1dIpdU6R00BHOyVcOMjE
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure data is in the correct format: long-form for seaborn
df_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Create the dot plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_long, x='year', y='percentage', hue='faculty_type', marker='o')

# Enhance the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 51 with threat_id: thread_HHXl7RBTVUJwVUjAbewdLrxL
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/path/to/your/file.csv'
data = pd.read_csv(file_path)

# Years and faculty types
years = data.columns[1:]  # Exclude the 'faculty_type' column
faculty_types = data['faculty_type']

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot each series as a separate line
for i, row in data.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for the legend
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 52 with threat_id: thread_BobNIxxyfuWdXeTyHxnusp4Y
import pandas as pd
import matplotlib.pyplot as plt

# Data preparation
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Set the years for the X-axis
years = df.columns[1:]  # Exclude 'faculty_type' 

# Plot setup
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c', 'm']
markers = ['o', 's', 'D', '^', 'v']

# Plot each faculty type
for i, row in df.iterrows():
    plt.scatter(years, row[1:], label=row['faculty_type'], color=colors[i % len(colors)], marker=markers[i % len(markers)])

# Plot details
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 53 with threat_id: thread_oUuXm5XzvTEB4dsQLYj9hfpD
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/path/to/your/file.csv'
df = pd.read_csv(file_path)

# Set the years as the columns for plotting
years = df.columns[1:]

# Create the dot plot
plt.figure(figsize=(10, 6))
for i, row in df.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 54 with threat_id: thread_7JBQ2JaowvtCFXWbmyNDuOd5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Melt the dataframe to make it suitable for seaborn plotting
melted_data = data.melt(id_vars="faculty_type", var_name="year", value_name="percentage")

# Convert years to integers for plotting
melted_data['year'] = melted_data['year'].astype(int)

# Plotting
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
dot_plot = sns.stripplot(x="year", y="percentage", hue="faculty_type", data=melted_data, dodge=True, jitter=True, marker='o', alpha=0.7)

# Customizing the plot
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14, Round 55 with threat_id: thread_IVxzReBQuR9IVSwAQZUikhf4
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate over each faculty type and plot its employment trend
for index, row in data.iterrows():
    faculty_data = row[1:]  # Exclude the faculty_type column
    ax.plot(faculty_data.index, faculty_data.values, 'o-', label=row['faculty_type'])

# Customize the plot
ax.set_title('Instructional Staff Employment Trends (1975-2011)')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_xticks(data.columns[1:])
ax.set_xticklabels(data.columns[1:], rotation=45)
ax.legend(title='Faculty Type')
ax.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 56 with threat_id: thread_XhC51FVhsmWSUopS5dAeyKHP
import matplotlib.pyplot as plt

# Transpose the data for easier plotting
data_transposed = data.set_index('faculty_type').T
data_transposed.index = data_transposed.index.astype(int)

# Initialize the plot
plt.figure(figsize=(12, 8))

# Iterate through each faculty type for plotting
for faculty in data_transposed.columns:
    plt.plot(data_transposed.index, data_transposed[faculty], 'o-', label=faculty)

# Adding labels and title
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 57 with threat_id: thread_sexImFcFaddSrQc82TsVCWle
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/path/to/your/data.csv'
data = pd.read_csv(file_path)

# Define years and faculty types
years = data.columns[1:].tolist()
faculty_types = data['faculty_type']

# Create Figure and Axes for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Dot plot
for index, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Add title, labels, and legend
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 58 with threat_id: thread_bQxSbu6ooVLL1iQk4DcqU7DK
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('path_to_your_file.csv')

# Melt the data for plotting
df_melted = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the dot plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 60 with threat_id: thread_JFyGKpdT0VNsrGy7oU7ElRgy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)
df_melted = df.melt(id_vars=["faculty_type"], var_name="year", value_name="percentage")

# Convert the year to an integer
df_melted['year'] = df_melted['year'].astype(int)

# Plotting
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Create the dot plot
sns.scatterplot(data=df_melted, x="year", y="percentage", hue="faculty_type", style="faculty_type", s=100)

# Add titled labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Faculty Type')
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 61 with threat_id: thread_05X86Uqmso795dBf3QTnSrGV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe for a long format suitable for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to a numeric format for plotting
data_melted['Year'] = data_melted['Year'].astype(int)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='Year', y='faculty_type', size='Percentage', legend=False, alpha=0.6, sizes=(20, 500))

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Faculty Type', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 62 with threat_id: thread_YxGQPW4y3PTu01NElm95XRFF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the style of the plot
sns.set(style="whitegrid")

# Convert the data from wide to long format for easier plotting
data_long = pd.melt(data, id_vars="faculty_type", var_name="Year", value_name="Proportion")

# Plot the dot plot
plt.figure(figsize=(10, 6))
sns.despine(left=True, bottom=True)
sns.scatterplot(data=data_long, x="Year", y="Proportion", hue="faculty_type", style="faculty_type", s=100, palette="muted")

# Add title and labels
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Proportion (%)")
plt.xticks(rotation=45)
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 63 with threat_id: thread_cXEhALqAtERxXtfXTRb2VRd0
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format
data_long = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Plotting
plt.figure(figsize=(14, 8))
sns.scatterplot(data=data_long, x='Year', y='faculty_type', size='Percentage', hue='faculty_type', legend=False, sizes=(20, 200))

# Title and labels
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Faculty Type', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(fontsize=10)

# Show plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 64 with threat_id: thread_RIJamSE79yQAVj9vFncIh3V5
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Melt the dataframe to have a long format suitable for dot plots
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert year to integer for proper sorting
data_melted['Year'] = data_melted['Year'].astype(int)

# Set the style
sns.set(style="whitegrid")

# Create dot plot
plt.figure(figsize=(10, 6))
dot_plot = sns.stripplot(x='Year', y='Percentage', hue='faculty_type', data=data_melted, dodge=True, jitter=0.1, marker='o', alpha=1, zorder=1)

# Adjusting plot parameters
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 65 with threat_id: thread_m9nD6iI1vPDP717kZfH6xk3z
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# Ensure you have the correct file path
file_path = '/path/to/your/datafile'  # Update this to your file path
data = pd.read_csv(file_path)

# Prepare the data for plotting
years = data.columns[1:]  # Extract the years from the columns
faculty_types = data['faculty_type']

# Plotting
plt.figure(figsize=(10, 6))

# Iterate through each faculty type and plot
for index, row in data.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Customizing the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage Employed')
plt.xticks(rotation=45)
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 66 with threat_id: thread_hV70tNMbAmCCJ4dkndOtKy5H
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transform the data from wide to long format
data_long = data.melt(id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Convert Year to a numeric type for plotting
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Plotting
plt.figure(figsize=(12, 8))

# Loop through each faculty type and plot
for faculty in data_long['faculty_type'].unique():
    subset = data_long[data_long['faculty_type'] == faculty]
    plt.plot(subset['Year'], subset['Percentage'], 'o-', label=faculty)

# Add labels and title
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Staff')
plt.legend(title='Faculty Type')
plt.grid(True)

# Display plot
plt.show()
##################################################
#Question 14, Round 67 with threat_id: thread_EzyOyGjSsHMCRzgQjOSossHf
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(14, 8))

# Extract years from the columns
years = data.columns[1:]

# Plot data for each faculty type
for index, row in data.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage of Total')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 68 with threat_id: thread_ys8aYCXipgBtcHSCXDqd86zb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/path/to/your/file.csv'  # Update this path to where your file is saved
data = pd.read_csv(file_path)

# Transform the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')
data_melted['year'] = data_melted['year'].astype(int)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 69 with threat_id: thread_eBhvcC9VQgEnCS84V0ic1dle
import matplotlib.pyplot as plt

# Transpose the data for easier plotting
data_transposed = data.set_index('faculty_type').transpose()

# Plot the data using a dot plot
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type
for faculty_type in data_transposed.columns:
    ax.plot(data_transposed.index, data_transposed[faculty_type], marker='o', label=faculty_type)

# Add title and labels
ax.set_title('Instructional Staff Employment Trends (1975-2011)')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')

# Rotate year labels
plt.xticks(rotation=45)

# Add a legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 70 with threat_id: thread_UWwEy5WX1qB5velX2VquFnz3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt data to long format for easier plotting with seaborn
melted_data = data.melt(id_vars='faculty_type', var_name='year', value_name='employment')

# Convert year to a numeric type
melted_data['year'] = melted_data['year'].astype(int)

# Set the size of the plot
plt.figure(figsize=(14, 8))

# Create a dot plot
sns.scatterplot(x='year', y='employment', hue='faculty_type', style='faculty_type',
                data=melted_data, s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the legend
plt.legend(title='Faculty Type')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 71 with threat_id: thread_0Y9CzqYgyyZM130d9LaoO9Fg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('your_file_path.csv')

# Melt the data for easier plotting with seaborn
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to integer for proper plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Set the plot size
plt.figure(figsize=(10, 6))

# Create the dot plot
sns.scatterplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', s=100, marker='o', palette='Set2')

# Add title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.legend(title='Faculty Type', loc='upper right')
plt.xticks(melted_data['Year'].unique())
plt.grid(True, linestyle='--', alpha=0.6)

# Show plot
plt.show()
##################################################
#Question 14, Round 72 with threat_id: thread_RF3OEuk3Pxx2oqCNqv9G2lms
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Transpose the dataset for plotting
data_melted = pd.melt(data, id_vars=['faculty_type'], var_name='year', value_name='percentage')

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 6))

# Create a dot plot
for key, grp in data_melted.groupby(['faculty_type']):
    ax.plot(grp['year'], grp['percentage'], marker='o', linestyle='', label=key)

# Adding labels and title
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 73 with threat_id: thread_22hZOje20DTxdXlmRE9JaURe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Convert data to long format for plotting
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14, Round 74 with threat_id: thread_tyz2gs78QOl5FvLsqhpiBZQM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Assuming the data contains columns like 'Year' and 'Employment', otherwise adjust as needed
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Year', y='Employment')
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment')
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data for plotting
melted_data = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Employment')

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=melted_data, x='Year', y='Employment', hue='faculty_type', style='faculty_type', s=100)
plt.title('Instructional Staff Employment Trends Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment (%)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 75 with threat_id: thread_u8Skd4IT5h1tbDkhG0Dnzyxv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from your file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the DataFrame for use with seaborn
data_melted = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Convert the 'Year' column to numeric
data_melted['Year'] = data_melted['Year'].astype(int)

# Create the dot plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize plot appearance
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 76 with threat_id: thread_xuKAqVHVr8xgD0LTBnuXD8VP
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to make it suitable for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert year to integer for proper sorting
data_melted['year'] = data_melted['year'].astype(int)

# Set the figure size
plt.figure(figsize=(10, 6))

# Iterate over each faculty type to create dots
for faculty in data['faculty_type'].unique():
    subset = data_melted[data_melted['faculty_type'] == faculty]
    plt.scatter(subset['year'], subset['percentage'], label=faculty, s=100)  # s=100 for larger dots

# Add titles and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')

# Show legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show grid
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 77 with threat_id: thread_nSr9lRMKbruDP2CDmpGlDpzl
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the CSV file is loaded into a DataFrame `data`
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melting the dataframe to have 'year' and 'percentage' columns
melted_data = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')
melted_data['year'] = melted_data['year'].astype(int)  # Ensure 'year' is an integer

# Plotting
plt.figure(figsize=(12, 8))
for faculty in data['faculty_type']:
    subset = melted_data[melted_data['faculty_type'] == faculty]
    plt.plot(subset['year'], subset['percentage'], marker='o', label=faculty)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(melted_data['year'].unique())  # Show all years on x-axis
plt.legend(title='Faculty Type')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 78 with threat_id: thread_YHEcngNA29Llz1fRuM2MXjY8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years as a variable
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 79 with threat_id: thread_3Q0j0C3yUYWczuBnE8yLuVsm
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('path_to_your_file.csv')

# Set up the plot
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']

# Plot each faculty type as a series in the dot plot
for i, row in data.iterrows():
    years = data.columns[1:]  # Excludes 'faculty_type' column
    percentages = row[1:]
    plt.plot(years, percentages, 'o-', label=row['faculty_type'], color=colors[i % len(colors)])

# Customize the plot
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Employment Percentage")
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 80 with threat_id: thread_raPbge1BNVEZS3KLpvSepckX
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe for easier plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to integer
melted_data['Year'] = melted_data['Year'].astype(int)

# Plot
plt.figure(figsize=(10, 6))
for faculty in data['faculty_type']:
    subset = melted_data[melted_data['faculty_type'] == faculty]
    plt.plot(subset['Year'], subset['Percentage'], 'o-', label=faculty)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(melted_data['Year'].unique(), rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 81 with threat_id: thread_I9Ag87kwLsceu8urwHb7jMC3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for easier plotting
data_melted = data.melt(id_vars=['faculty_type'], var_name='Year', value_name='Employment')

# Create a dot plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
dot_plot = sns.scatterplot(data=data_melted, x='Year', y='Employment', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.show()
##################################################
#Question 14, Round 82 with threat_id: thread_E4clky7n4BQ0n3MwBhi5F3fS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Employment')

# Convert the "Year" column to integers
data_melted['Year'] = data_melted['Year'].astype(int)

# Create a dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='Year', y='faculty_type', size='Employment', legend=False, sizes=(20, 400))

# Add labels and title
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Faculty Type')
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 83 with threat_id: thread_RZhsjuWaMcWi2LPGQDrEAg9p
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/path/to/your/file.csv'
data = pd.read_csv(file_path)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Extract years from the columns, assuming they start from the second column
years = data.columns[1:]

# Plot each faculty type
for index, row in data.iterrows():
    ax.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Customize the plot
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)
ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 84 with threat_id: thread_2ckS9xArh0els5k1x7dCjeU2
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Convert the data to a long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a dot plot
plt.figure(figsize=(10, 8))

# Plot each faculty type
for faculty_type in data['faculty_type']:
    faculty_data = data_long[data_long['faculty_type'] == faculty_type]
    plt.plot(faculty_data['Year'], faculty_data['Percentage'], 'o', label=faculty_type)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.grid(True)
plt.show()
##################################################
#Question 14, Round 85 with threat_id: thread_15wf5f2jgPFlRWICHRYmtvQq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'path_to_your_file.csv'  # Update this with the correct path
data = pd.read_csv(file_path)

# Melt the data for easier plotting
df_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to numeric for plotting
df_melted['Year'] = pd.to_numeric(df_melted['Year'])

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
##################################################
#Question 14, Round 86 with threat_id: thread_lTusHACAlra36uUZPELm9oFr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Set up the plot dimensions
plt.figure(figsize=(12, 8))

# Transform the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
for faculty in data['faculty_type']:
    subset = data_melted[data_melted['faculty_type'] == faculty]
    plt.plot(subset['year'], subset['percentage'], 'o-', label=faculty)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 50, 5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 87 with threat_id: thread_qPMdkUVqeQjlWuOMnc9Sv9Dc
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Plot
plt.figure(figsize=(10, 6))
for key, grp in data_melted.groupby(['faculty_type']):
    plt.plot(grp['Year'], grp['Percentage'], marker='o', linestyle='', label=key)

# Adding Labels and Title
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.grid(True)

# Display the plot
plt.show()
##################################################
#Question 14, Round 88 with threat_id: thread_ffMs9G2V2pK73bOQTa3pApUL
import matplotlib.pyplot as plt

# Transpose the data for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Plotting
plt.figure(figsize=(12, 8))
for label, df in data_long.groupby('faculty_type'):
    plt.plot(df['Year'], df['Percentage'], 'o-', label=label)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
##################################################
#Question 14, Round 89 with threat_id: thread_6eyk9wpYbmCX0AXI8Iv4DuX2
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format suitable for plotting
data_long = data.melt(id_vars=['faculty_type'], var_name='year', value_name='percentage')

# Convert the 'year' column to integer
data_long['year'] = data_long['year'].astype(int)

# Plotting
plt.figure(figsize=(12, 8))

for key, grp in data_long.groupby(['faculty_type']):
    plt.plot(grp['year'], grp['percentage'], marker='o', linestyle='', markersize=6, label=key)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(data_long['year'].unique())
plt.legend(title='Faculty Type')
plt.show()
##################################################
#Question 14, Round 90 with threat_id: thread_A4IIdJJkczuCUFIX4z3kM1ko
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for easier plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to numeric for plotting
melted_data['Year'] = pd.to_numeric(melted_data['Year'])

# Set the plot style
sns.set(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', markers=True, dashes=False)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14, Round 91 with threat_id: thread_lIvIhgkYQd0JvuW5QU9PDGBi
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/path/to/your/file.csv')

# Set the figure size
plt.figure(figsize=(10, 6))

# Iterate over each faculty type and plot
for index, row in data.iterrows():
    years = row.index[1:]
    values = row.values[1:]
    plt.plot(years, values, 'o-', label=row['faculty_type'])  # Line with dots

# Add labels and title
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14, Round 92 with threat_id: thread_Xf6W18vg2FTiSXK7v8mSVRw6
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Set up the figure and axes
plt.figure(figsize=(12, 8))

# For each faculty type, plot the data
for index, row in data.iterrows():
    plt.plot(data.columns[1:], row[1:], 'o-', label=row['faculty_type'])

# Add labels and title
plt.title('Instructional Staff Employment Trends', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 93 with threat_id: thread_C9gPKbDbyIQ98cguA58oBCCK
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/path/to/your/file.csv')

# Set the plot style
sns.set(style="whitegrid")

# Transform the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', marker="o")

# Set plot labels and title
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)

# Show the plot
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 94 with threat_id: thread_LWvCwy03Fiful0ZhyWk9fK5S
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data.csv' is the file containing the provided CSV data
data = pd.read_csv('your_file.csv', encoding='utf-8-sig')

# Prepare the data for plotting
years = data.columns[1:]  # Extract year columns
faculty_types = data['faculty_type']

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each faculty type
for idx, row in data.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'], markersize=8)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14, Round 95 with threat_id: thread_dT11cMZ3OSrwaZHuDxBB4atb
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/path/to/your/file.csv'
data = pd.read_csv(file_path)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each faculty type as a separate series
for index, row in data.iterrows():
    plt.plot(data.columns[1:], row[1:], marker='o', label=row['faculty_type'])

# Add titles and labels
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Instructional Staff')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

# Add a legend
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 96 with threat_id: thread_jv67BqJ1T6TItANyHQWJHib0
import pandas as pd
import matplotlib.pyplot as plt

# Data preparation
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')
data.set_index("faculty_type", inplace=True)
data_transposed = data.T

# Plot
plt.figure(figsize=(10, 6))
for column in data_transposed.columns:
    plt.plot(data_transposed.index, data_transposed[column], 'o-', label=column)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 97 with threat_id: thread_qIJWYHqFKMrfhO4IHBFPZ4ea
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(10, 6))

# Iterate over each row in the data to plot each faculty type
for index, row in data.iterrows():
    plt.plot(row.index[1:], row.values[1:], 'o-', label=row['faculty_type'])

# Add plot titles and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14, Round 98 with threat_id: thread_dbSAdwcEpVWoioIOVCfjDlQe
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (replace with your file path)
file_path = '/path/to/your/file.csv'
data = pd.read_csv(file_path)

# Transpose the data for plotting purposes
data_transposed = data.set_index('faculty_type').transpose()

# Create a dot plot
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over each row in the DataFrame
for index, row in data_transposed.iterrows():
    ax.plot(row.index, row.values, 'o-', label=index)

# Labeling the plot
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14, Round 99 with threat_id: thread_FlC1nGR2lOgqBUMZMd206ruo
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Extract years from the column names
years = data.columns[1:].astype(int)

# Loop through each faculty type and plot the dots
for index, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show grid
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to make room for the legend
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 0 with threat_id: thread_xb7Ca7r69XNBvqIRD6ontG6I
import matplotlib.pyplot as plt

# Transpose dataframe to make plotting easier
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create dot plot
plt.figure(figsize=(10, 6))

# Plot the data
for faculty_type, group_data in data_long.groupby('faculty_type'):
    plt.plot(group_data['Year'], group_data['Percentage'], 'o-', label=faculty_type)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


import numpy as np

# Reset index for easier plotting
data.set_index('faculty_type', inplace=True)

# Create a new figure
plt.figure(figsize=(12, 8))

# Bar plot for different years
n_years = len(data.columns)
bar_width = 0.15
index = np.arange(len(data.index))

# Plot each year as a separate bar cluster 
for i, year in enumerate(data.columns):
    plt.bar(index + i * bar_width, data[year], bar_width, label=year)

# Customize plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Faculty Type')
plt.ylabel('Employment Percentage')
plt.xticks(index + (n_years - 1) * bar_width / 2, data.index, rotation=45)
plt.legend(title='Year', loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(axis='y')
plt.tight_layout()

# Show plot
plt.show()


# Transpose dataframe for easier manipulation
data.reset_index(inplace=True)

# Create an improved version of the bar plot
plt.figure(figsize=(14, 8))

# Customize bar plot colors and styles
colors = plt.cm.viridis(np.linspace(0, 1, n_years))

# Create stacked bars for easier comparison over years
bottoms = np.zeros(len(data.index))
for i, (year, color) in enumerate(zip(data.columns[1:], colors)):
    plt.bar(data['faculty_type'], data[year], label=year, color=color, bottom=bottoms)
    bottoms += data[year]

# Adjust plot aesthetics
plt.title('Enhanced Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Faculty Type', fontsize=12)
plt.ylabel('Cumulative Employment Percentage', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Year', title_fontsize=12, fontsize=10)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()

# Show the enhanced plot
plt.show()
##################################################
#Question 14.2, Round 1 with threat_id: thread_UUjAMrMLclTdhZ65rb6pnBKt
import matplotlib.pyplot as plt

# Prepare the data for plotting
years = data.columns[1:]  # Exclude the first column which is 'faculty_type'
x_values = range(len(years))  # X-coordinates for the plot

# Plotting
plt.figure(figsize=(12, 8))
for index, row in data.iterrows():
    plt.fill_between(x_values, row[1:], label=row['faculty_type'], alpha=0.5)

# Adding the labels and title
plt.xticks(x_values, years, rotation=45)
plt.title('Instructional Staff Employment Trends (Filled Area Plot)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 4 with threat_id: thread_CldtFCsAgGGpAMCp8p1XlbxE
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/path/to/your/file.csv'
data = pd.read_csv(file_path)

# Set the plot size for better readability
plt.figure(figsize=(12, 8))

# Plot each faculty type over the years
for index, row in data.iterrows():
    plt.plot(data.columns[1:], row[1:], marker='o', label=row['faculty_type'])

# Adding title and labels
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage', fontsize=12)

# Add a legend
plt.legend(title='Faculty Type', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 6 with threat_id: thread_lKlYBhCfEJuYac6eF2ZDAvgP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV'
data = pd.read_csv(file_path)

# Prepare the data for plotting
data_melted = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Refine the plot with improved labels and aesthetics
plt.figure(figsize=(12, 7))
sns.lineplot(data=data_melted, x="Year", y="Percentage", hue="faculty_type", marker='o', linewidth=2.5)

# Enhancing labels and title
plt.title("Trends in Instructional Staff Employment (1975-2011)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Percentage of Total Employment (%)", fontsize=12)

# Enhancing legend
plt.legend(title="Type of Faculty", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize='13')

# Grid and ticks customization
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 7 with threat_id: thread_uMCX28Bsk3ffWPhhEm9wbwLM
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV')

# Transpose the data for plotting
data_transposed = data.set_index('faculty_type').T

# Create the dot plot
fig, ax = plt.subplots(figsize=(10, 6))

for category in data_transposed.columns:
    ax.plot(data_transposed.index, data_transposed[category], 'o-', label=category)

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Employment')
ax.set_title('Instructional Staff Employment Trends (Dot Plot)')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV')

# Transpose the data for plotting
data_transposed = data.set_index('faculty_type').T

# Create the line plot
fig, ax = plt.subplots(figsize=(12, 8))

for category in data_transposed.columns:
    ax.plot(data_transposed.index, data_transposed[category], marker='o', linestyle='-', label=category)

# Add labels and title with improvements
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Percentage Employment', fontsize=12)
ax.set_title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
ax.legend(title='Faculty Type', title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 8 with threat_id: thread_pRukXdRiIvhWJs2MfRAlcpOt
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV'
data = pd.read_csv(file_path)

# Transpose for convenience
data_transposed = data.melt(id_vars=["faculty_type"], var_name="year", value_name="percentage")

# Creating the dot plot
plt.figure(figsize=(10, 6))
for label, df in data_transposed.groupby('faculty_type'):
    plt.plot(df['year'], df['percentage'], 'o', label=label)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 13 with threat_id: thread_NVsFke1DpQykZOgGFz2eKAl4
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV'
data = pd.read_csv(file_path)

# Prepare data for the area plot by transposing and converting to numeric
transposed_data = data.set_index('faculty_type').T
transposed_data.index = pd.to_numeric(transposed_data.index)
years_numeric = transposed_data.index

# Adjust the color list to match the number of faculty types
colors_corrected = ['b', 'g', 'r', 'c', 'm']

# Plot a stacked area plot
plt.figure(figsize=(12, 7))
plt.stackplot(
    years_numeric,
    transposed_data.T,
    labels=transposed_data.columns,
    colors=colors_corrected
)

# Add improvements: title, axis labels and legend
plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Faculty')
plt.legend(title='Faculty Type', loc='upper left')

# Show the plot
plt.grid(True)
plt.xticks(years_numeric, rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 14 with threat_id: thread_7gutLGWj1EBQyL5SuVdDEOql
# Step 1: Read and display the content of the uploaded file
import pandas as pd

# Load the data
file_path = '/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()


import matplotlib.pyplot as plt

# Step 2: Prepare data for plotting
years = data.columns[1:]  # Extract years from columns, ignoring the first "faculty_type" column
faculty_types = data['faculty_type']  # Extract faculty types

# Step 3: Create a line plot
plt.figure(figsize=(10, 6))
for i in range(len(data)):
    plt.plot(years, data.iloc[i, 1:], label=faculty_types[i])

# Step 4: Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Proportion of Total Faculty (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Step 5: Show the plot
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Prepare data for plotting
years = data.columns[1:]  # Extract years from columns, ignoring the first "faculty_type" column
faculty_types = data['faculty_type']  # Extract faculty types

# Create a line plot
plt.figure(figsize=(10, 6))
for i in range(len(data)):
    plt.plot(years, data.iloc[i, 1:], marker='o', label=faculty_types[i])

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Proportion of Total Faculty (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 17 with threat_id: thread_Hd90LS8bbQW7GJoiQpDvLFs7
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
file_path = '/mnt/data/file-XtRrHxSoqJsT7Jr1Xw1bUV'
data = pd.read_csv(file_path)

# Define a function to plot the instructional staff employment trends
def plot_employment_trends(data):
    plt.figure(figsize=(12, 8))

    for index, row in data.iterrows():
        plt.plot(data.columns[1:], row[1:], marker='o', label=row['faculty_type'])

    plt.title('Instructional Staff Employment Trends (1975-2011)')
    plt.xlabel('Year')
    plt.ylabel('Employment Percentage')
    plt.xticks(rotation=45)
    plt.yticks(range(0, 45, 5))
    plt.legend(title='Faculty Type')
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()

# Call the function to plot the data
plot_employment_trends(data)
##################################################
#Question 14.0, Round 20 with threat_id: thread_AjBdqDAKdl61iAtEUrmfIGmC
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_transposed = data.set_index('faculty_type').T
years = data_transposed.index.astype(int)  # Convert years to integer for plotting

# Plotting the dot plot
plt.figure(figsize=(10, 6))

# Plot each faculty type
for faculty_type in data['faculty_type']:
    plt.plot(years, data_transposed[faculty_type], 'o', label=faculty_type)

plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(years, rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 20 with threat_id: thread_AjBdqDAKdl61iAtEUrmfIGmC
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_transposed = data.set_index('faculty_type').T
years = data_transposed.index.astype(int)  # Convert years to integer for plotting

# Plotting the line plot with markers
plt.figure(figsize=(10, 6))

# Plot each faculty type
for faculty_type in data['faculty_type']:
    plt.plot(years, data_transposed[faculty_type], marker='o', linestyle='-', label=faculty_type)

plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(years, rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 20 with threat_id: thread_AjBdqDAKdl61iAtEUrmfIGmC
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_transposed = data.set_index('faculty_type').T
years = data_transposed.index.astype(int)  # Convert years to integer for plotting

# Plotting the line plot with improved styling
plt.figure(figsize=(12, 7))

# Define a color palette for variety in the plot
colors = plt.cm.viridis_r(range(len(data['faculty_type'])))

# Plot each faculty type
for i, faculty_type in enumerate(data['faculty_type']):
    plt.plot(years, data_transposed[faculty_type], marker='o', linestyle='-', label=faculty_type, color=colors[i])

plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Faculty Employment', fontsize=12)
plt.xticks(years, rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize='11', frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 21 with threat_id: thread_uWtTXBmIFEOmwLJ0JVVqEAmL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format
long_data = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert the years to integers for sorting
long_data['year'] = long_data['year'].astype(int)

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(data=long_data, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Add titles and labels
plt.title('Instructional Staff Employment Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(long_data['year'].unique())  # Ensure all years are displayed
plt.legend(title='Faculty Type')
plt.grid(True)

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 21 with threat_id: thread_uWtTXBmIFEOmwLJ0JVVqEAmL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format
long_data = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert the years to integers for sorting
long_data['year'] = long_data['year'].astype(int)

# Set the style for the plot
sns.set(style='whitegrid')

# Create a line plot with markers
plt.figure(figsize=(12, 8))
sns.lineplot(data=long_data, x='year', y='percentage', hue='faculty_type', marker='o')

# Add titles and labels
plt.title('Instructional Staff Employment Trends with Line Plot')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(long_data['year'].unique())  # Ensure all years are displayed
plt.legend(title='Faculty Type')
plt.grid(True)

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 21 with threat_id: thread_uWtTXBmIFEOmwLJ0JVVqEAmL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format
long_data = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert the years to integers for sorting
long_data['year'] = long_data['year'].astype(int)

# Set the style for the plot
sns.set(style='whitegrid')

# Create a line plot with markers
plt.figure(figsize=(14, 8))
sns.lineplot(data=long_data, x='year', y='percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Add titles and labels with improved clarity
plt.title('Trends in Employment of Instructional Staff Over Time', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.xticks(long_data['year'].unique(), fontsize=12)
plt.yticks(fontsize=12)

# Improve the legend
plt.legend(title='Faculty Category', title_fontsize='13', fontsize='11', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', linewidth=0.7)

# Show the plot
plt.tight_layout()  # Adjust the layout to make room for the legend
plt.show()
##################################################
#Question 14.0, Round 22 with threat_id: thread_74wf0n0vvzUhlzpYoK1baPaC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transform the data to a long format for plotting
data_long = pd.melt(data, id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x="Year", y="Percentage", hue="faculty_type", style="faculty_type", s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975 - 2011)')
plt.ylabel('Percentage of Employment')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 22 with threat_id: thread_74wf0n0vvzUhlzpYoK1baPaC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transform the data to a long format for plotting
data_long = pd.melt(data, id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Plotting
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_long, x="Year", y="Percentage", hue="faculty_type", marker="o")

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975 - 2011)')
plt.ylabel('Percentage of Employment')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 22 with threat_id: thread_74wf0n0vvzUhlzpYoK1baPaC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transform the data to a long format for plotting
data_long = pd.melt(data, id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Plotting
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_long, x="Year", y="Percentage", hue="faculty_type", marker="o")

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975 - 2011)', fontsize=16)
plt.ylabel('Employment Percentage (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize='11')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 23 with threat_id: thread_sJ47cvCmtprUnVmwJ6fXHDA9
import matplotlib.pyplot as plt

# Transpose the data to have years as rows for plotting
data_melted = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Convert 'Year' to integer for sorting and plotting
data_melted['Year'] = data_melted['Year'].astype(int)

# Create the dot plot
plt.figure(figsize=(10, 6))
for faculty_type in data['faculty_type']:
    subset = data_melted[data_melted['faculty_type'] == faculty_type]
    plt.plot(subset['Year'], subset['Percentage'], 'o-', label=faculty_type)

# Add titles and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.1, Round 23 with threat_id: thread_sJ47cvCmtprUnVmwJ6fXHDA9
import matplotlib.pyplot as plt

# Set up data for stacked area plot
years = data.columns[1:].astype(int)  # Extract the years as integers
faculty_types = data['faculty_type']
values = data.iloc[:, 1:].values  # Extract the employment percentages as a 2D array

# Create the stacked area plot
plt.figure(figsize=(10, 6))
plt.stackplot(years, *values, labels=faculty_types, alpha=0.8)

# Add titles and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 23 with threat_id: thread_sJ47cvCmtprUnVmwJ6fXHDA9
import matplotlib.pyplot as plt

# Set up data for stacked area plot
years = data.columns[1:].astype(int)  # Extract the years as integers
faculty_types = data['faculty_type']
values = data.iloc[:, 1:].values  # Extract the employment percentages as a 2D array

# Create the stacked area plot
plt.figure(figsize=(12, 7))
plt.stackplot(years, *values, labels=faculty_types, alpha=0.8)

# Add titles and labels with improved styling
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Enhance the legend
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)  # Adjust grid aesthetics
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 24 with threat_id: thread_t3eRTYGGGBtihyqSOrVjgSl5
import matplotlib.pyplot as plt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type as a series of dots
years = data.columns[1:]  # Extract year columns
for index, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage Employment')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show grid
ax.grid(True)

# Adjust the subplot to fit legend
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 24 with threat_id: thread_t3eRTYGGGBtihyqSOrVjgSl5
import matplotlib.pyplot as plt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type as a line
years = data.columns[1:]  # Extract year columns
for index, row in data.iterrows():
    ax.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage Employment')
ax.set_title('Instructional Staff Employment Trends (Line Plot)')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show grid
ax.grid(True)

# Adjust the subplot to fit the legend
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 24 with threat_id: thread_t3eRTYGGGBtihyqSOrVjgSl5
import matplotlib.pyplot as plt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each faculty type as a line with distinct markers and colors
years = data.columns[1:]  # Extract year columns
for index, row in data.iterrows():
    ax.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Add labels and title with improved styling
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of Employment (%)', fontsize=12, fontweight='bold')
ax.set_title('Trends in Instructional Staff Employment from 1975 to 2011', fontsize=14, fontweight='bold')

# Customize legend for better readability
ax.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)

# Show grid with improved styling
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Adjust figure layout to accommodate legend
plt.tight_layout()

# Improve axis tick size for readability
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 26 with threat_id: thread_2GGrgJTJ274Dg3ehzN1d1pbr
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.DataFrame({
    'faculty_type': ['Full-Time Tenured Faculty', 'Full-Time Tenure-Track Faculty', 'Full-Time Non-Tenure-Track Faculty',
                     'Part-Time Faculty', 'Graduate Student Employees'],
    '1975': [29.0, 16.1, 10.3, 24.0, 20.5],
    '1989': [27.6, 11.4, 14.1, 30.4, 16.5],
    '1993': [25.0, 10.2, 13.6, 33.1, 18.1],
    '1995': [24.8, 9.6, 13.6, 33.2, 18.8],
    '1999': [21.8, 8.9, 15.2, 35.5, 18.7],
    '2001': [20.3, 9.2, 15.5, 36.0, 19.0],
    '2003': [19.3, 8.8, 15.0, 37.0, 20.0],
    '2005': [17.8, 8.2, 14.8, 39.3, 19.9],
    '2007': [17.2, 8.0, 14.9, 40.5, 19.5],
    '2009': [16.8, 7.6, 15.1, 41.1, 19.4],
    '2011': [16.7, 7.4, 15.4, 41.3, 19.3]
})

# Melt the DataFrame to "long-form" or "tidy" representation
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Create the dot plot
plt.figure(figsize=(10, 6))
for ft in data_melted['faculty_type'].unique():
    subset = data_melted[data_melted['faculty_type'] == ft]
    plt.plot(subset['year'], subset['percentage'], 'o-', label=ft)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 26 with threat_id: thread_2GGrgJTJ274Dg3ehzN1d1pbr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
data = pd.DataFrame({
    'faculty_type': ['Full-Time Tenured Faculty', 'Full-Time Tenure-Track Faculty', 'Full-Time Non-Tenure-Track Faculty',
                     'Part-Time Faculty', 'Graduate Student Employees'],
    '1975': [29.0, 16.1, 10.3, 24.0, 20.5],
    '1989': [27.6, 11.4, 14.1, 30.4, 16.5],
    '1993': [25.0, 10.2, 13.6, 33.1, 18.1],
    '1995': [24.8, 9.6, 13.6, 33.2, 18.8],
    '1999': [21.8, 8.9, 15.2, 35.5, 18.7],
    '2001': [20.3, 9.2, 15.5, 36.0, 19.0],
    '2003': [19.3, 8.8, 15.0, 37.0, 20.0],
    '2005': [17.8, 8.2, 14.8, 39.3, 19.9],
    '2007': [17.2, 8.0, 14.9, 40.5, 19.5],
    '2009': [16.8, 7.6, 15.1, 41.1, 19.4],
    '2011': [16.7, 7.4, 15.4, 41.3, 19.3]
})

# Melt the DataFrame to "long-form" or "tidy" representation
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Create the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(data=data_melted, x='year', y='percentage', hue='faculty_type')

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 26 with threat_id: thread_2GGrgJTJ274Dg3ehzN1d1pbr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
data = pd.DataFrame({
    'faculty_type': ['Full-Time Tenured Faculty', 'Full-Time Tenure-Track Faculty', 'Full-Time Non-Tenure-Track Faculty',
                     'Part-Time Faculty', 'Graduate Student Employees'],
    '1975': [29.0, 16.1, 10.3, 24.0, 20.5],
    '1989': [27.6, 11.4, 14.1, 30.4, 16.5],
    '1993': [25.0, 10.2, 13.6, 33.1, 18.1],
    '1995': [24.8, 9.6, 13.6, 33.2, 18.8],
    '1999': [21.8, 8.9, 15.2, 35.5, 18.7],
    '2001': [20.3, 9.2, 15.5, 36.0, 19.0],
    '2003': [19.3, 8.8, 15.0, 37.0, 20.0],
    '2005': [17.8, 8.2, 14.8, 39.3, 19.9],
    '2007': [17.2, 8.0, 14.9, 40.5, 19.5],
    '2009': [16.8, 7.6, 15.1, 41.1, 19.4],
    '2011': [16.7, 7.4, 15.4, 41.3, 19.3]
})

# Melt the DataFrame to "long-form" or "tidy" representation
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Set the style and palette
sns.set(style='whitegrid')
palette = sns.color_palette('pastel', n_colors=len(data['faculty_type'].unique()))

# Create the bar plot
plt.figure(figsize=(14, 8))
sns.barplot(data=data_melted, x='year', y='percentage', hue='faculty_type', palette=palette)

# Improve the plot's aesthetics
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='13', fontsize='11')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 28 with threat_id: thread_6EoucxKdnQKR0KM2YDtEIGXl
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare the data again, if necessary
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot dimensions and style
plt.figure(figsize=(14, 8))

# Create the area plot with improved aesthetics
data.set_index('faculty_type').T.plot(kind='area', stacked=True, alpha=0.85, colormap='viridis')

# Enhance the plot with improved labels, title, and legend
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, weight='bold')
plt.xlabel('Year', fontsize=12, weight='bold')
plt.ylabel('Percentage of Total Employment (%)', fontsize=12, weight='bold')
plt.xticks(rotation=45)
plt.legend(loc='upper right', title='Faculty Type', fontsize=10, title_fontsize='12')
plt.grid(True, linestyle='--', alpha=0.7)

# Add annotations for additional context
for year in [1975, 1995, 2011]:
    plt.axvline(x=year, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.text(year, 80, str(year), fontsize=9, color='black', ha='center')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 29 with threat_id: thread_GqfXLoxoWPqRCBV5S57sbh2K
import pandas as pd

# Load the contents of the uploaded file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()


import matplotlib.pyplot as plt

# Extract years and faculty types for plotting
years = data.columns[1:].astype(int)  # Years (as int)
faculty_types = data['faculty_type']  # Faculty types

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type's trend
for index, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Add labels and title
ax.set_xlabel("Year")
ax.set_ylabel("Percentage of Total Instructional Staff")
ax.set_title("Instructional Staff Employment Trends (1975-2011)")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside the plot

# Show grid
ax.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 29 with threat_id: thread_GqfXLoxoWPqRCBV5S57sbh2K
import matplotlib.pyplot as plt

# Extract years and faculty types for plotting
years = data.columns[1:].astype(int)  # Years (as int)
faculty_types = data['faculty_type']  # Faculty types

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type's trend using a line plot with filled areas
for index, row in data.iterrows():
    ax.fill_between(years, 0, row[1:], label=row['faculty_type'], alpha=0.5)

# Add labels and title
ax.set_xlabel("Year")
ax.set_ylabel("Percentage of Total Instructional Staff")
ax.set_title("Instructional Staff Employment Trends (1975-2011)")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside the plot

# Show grid
ax.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 29 with threat_id: thread_GqfXLoxoWPqRCBV5S57sbh2K
import matplotlib.pyplot as plt
import seaborn as sns

# Set theme for seaborn for better aesthetics
sns.set_theme(style="whitegrid")

# Extract years and faculty types for plotting
years = data.columns[1:].astype(int)  # Years (as int)
faculty_types = data['faculty_type']  # Faculty types

# Define a color palette
palette = sns.color_palette("husl", len(faculty_types))

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each faculty type's trend using a filled line plot
for index, (row, color) in enumerate(zip(data.iterrows(), palette)):
    fac_data = row[1]
    ax.fill_between(years, 0, fac_data[1:], label=fac_data['faculty_type'], alpha=0.6, color=color, linewidth=2)

# Add improved labels and title
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Percentage of Total Instructional Staff", fontsize=12)
ax.set_title("Instructional Staff Employment Trends (1975-2011)", fontsize=14, weight='bold')

# Move legend outside the plot for better clarity
ax.legend(title="Faculty Type", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title_fontsize=12)

# Tighten the layout for better fit
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate the legend outside

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 30 with threat_id: thread_QYmkFQxwGQaeXi9PYEJ92EEa
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Prepare the data for plotting
years = data.columns[1:].astype(int)  # Extract year columns
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each faculty type trend as a dot plot
for index, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Customize the plot
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)

# Display the plot
plt.xticks(years)  # Set x-ticks to be exactly the years present
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate legend
plt.show()
##################################################
#Question 14.1, Round 30 with threat_id: thread_QYmkFQxwGQaeXi9PYEJ92EEa
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Prepare data for plotting
years = data.columns[1:].astype(int)  # Extract year columns
employment_data = data.drop('faculty_type', axis=1).set_index(years).transpose()

# Create a stacked area plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.stackplot(years, employment_data.transpose(), labels=data['faculty_type'], alpha=0.8)

# Customize the plot
ax.set_title('Instructional Staff Employment Trends - Stacked Area Plot')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)

# Display the plot
plt.xticks(years)  # Set x-ticks to be exactly the years present
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate legend
plt.show()
##################################################
#Question 14.2, Round 30 with threat_id: thread_QYmkFQxwGQaeXi9PYEJ92EEa
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Prepare data for plotting
years = data.columns[1:].astype(int)  # Extract year columns
employment_data = data.drop('faculty_type', axis=1).set_index(years).transpose()

# Create a stacked area plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.stackplot(years, employment_data.transpose(), labels=data['faculty_type'], alpha=0.8)

# Customize the plot
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Employment Percentage (%)', fontsize=12)
ax.legend(title='Faculty Types', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize='11')
ax.grid(True, linestyle='--', alpha=0.7)

# Enhance axis ticks and grid
ax.set_xticks(years)
ax.set_yticks(range(0, 101, 10))
ax.set_xlim(min(years), max(years))

# Display the plot
plt.xticks(rotation=45)  # Rotate x-ticks for better readability
plt.tight_layout(rect=[0, 0, 0.88, 1])  # Adjust layout to accommodate legend
plt.show()
##################################################
#Question 14.0, Round 31 with threat_id: thread_pbmUUApEJ5TI3VjjUhEj61Qk
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the size of the plot
plt.figure(figsize=(14, 8))

# Plot a dot plot for each faculty type over time
for index, row in data.iterrows():
    plt.plot(data.columns[1:], row[1:], marker='o', label=row['faculty_type'])

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends (1975 - 2011)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')

# Show grid
plt.grid(True)

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 31 with threat_id: thread_pbmUUApEJ5TI3VjjUhEj61Qk
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting with grouped bars
data_transposed = data.set_index('faculty_type').T

# Set the size of the plot
plt.figure(figsize=(14, 8))

# Plot a bar plot for each year, with bars grouped by faculty type
data_transposed.plot(kind='bar', stacked=True, figsize=(14, 8))

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends (1975 - 2011)')
plt.xticks(rotation=45)

# Show legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 31 with threat_id: thread_pbmUUApEJ5TI3VjjUhEj61Qk
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting with grouped bars
data_transposed = data.set_index('faculty_type').T

# Set the size of the plot
plt.figure(figsize=(14, 8))

# Plot a bar plot for each year, with bars grouped by faculty type
data_transposed.plot(kind='bar', stacked=True, figsize=(14, 8), width=0.8, cmap='tab20')

# Add labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Faculty', fontsize=12)
plt.title('Trends in Instructional Staff Employment (1975 - 2011)', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Adjust legend positioning and title
plt.legend(title='Faculty Type', title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='10')

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 32 with threat_id: thread_pl1FfZjdKiw9zwSYQqzUaWg9
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the years relevant to the data
years = list(map(str, range(1975, 2012, 2)))

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type's employment trend as a dot plot
for index, row in data.iterrows():
    ax.plot(years, row[years], 'o-', label=row['faculty_type'])

# Set the title, labels, and legend
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 32 with threat_id: thread_pl1FfZjdKiw9zwSYQqzUaWg9
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the years relevant to the data
years = list(map(str, range(1975, 2012, 2)))

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for the stacked area plot
area_data = data.set_index('faculty_type')[years].T

# Plot each faculty type's employment trend as a stacked area plot
area_data.plot(kind='area', ax=ax, stacked=True, alpha=0.8)

# Set the title, labels, and legend
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 32 with threat_id: thread_pl1FfZjdKiw9zwSYQqzUaWg9
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the years relevant to the data
years = list(map(str, range(1975, 2012, 2)))

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data for the stacked area plot
area_data = data.set_index('faculty_type')[years].T

# Choose a color palette
colormap = cm.get_cmap('tab20')

# Plot each faculty type's employment trend as a stacked area plot
area_data.plot(kind='area', ax=ax, stacked=True, alpha=0.85, cmap=colormap)

# Set the title and labels with improved clarity
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Employment Percentage (%)', fontsize=14)

# Customizing the legend
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
legend = ax.get_legend()
legend.set_title('Faculty Type')
plt.setp(legend.get_texts(), fontweight='bold')

# Customize ticks for better readability
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.5)

# Adjust layout for better viewing
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 33 with threat_id: thread_WAXRq6k3ugqkI3m19bW2Hzze
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/path/to/your/file.csv' # Update this with your file path
data = pd.read_csv(file_path)

# Set plot style
plt.style.use('seaborn-whitegrid')

# Extract years from the columns (excluding 'faculty_type')
years = data.columns[1:]

# Transpose the DataFrame for easier plotting
data_melted = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a dot plot
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over each faculty type and plot
for faculty in data['faculty_type']:
    subset = data_melted[data_melted['faculty_type'] == faculty]
    ax.plot(subset['Year'], subset['Percentage'], label=faculty, marker='o')

# Customizing the plot
ax.set_title("Instructional Staff Employment Trends")
ax.set_xlabel("Year")
ax.set_ylabel("Percentage")
ax.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 33 with threat_id: thread_WAXRq6k3ugqkI3m19bW2Hzze
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/path/to/your/file.csv'  # Update this with your file path
data = pd.read_csv(file_path)

# Set plot style
plt.style.use('ggplot')

# Extract years from the columns (excluding 'faculty_type')
years = data.columns[1:]

# Transpose the DataFrame for easier plotting
data_melted = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a line plot with markers
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over each faculty type and plot
for faculty in data['faculty_type']:
    subset = data_melted[data_melted['faculty_type'] == faculty]
    ax.plot(subset['Year'], subset['Percentage'], label=faculty, marker='o')

# Customizing the plot
ax.set_title("Instructional Staff Employment Trends")
ax.set_xlabel("Year")
ax.set_ylabel("Percentage")
ax.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 33 with threat_id: thread_WAXRq6k3ugqkI3m19bW2Hzze
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/path/to/your/file.csv'  # Update this with your file path
data = pd.read_csv(file_path)

# Set plot style
sns.set(style="whitegrid")

# Extract years from the columns (excluding 'faculty_type')
years = data.columns[1:]

# Transpose the DataFrame for easier plotting
data_melted = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a color palette
palette = sns.color_palette("husl", len(data['faculty_type'].unique()))

# Create a line plot with markers
fig, ax = plt.subplots(figsize=(12, 8))

# Iterate over each faculty type and plot
for idx, faculty in enumerate(data['faculty_type']):
    subset = data_melted[data_melted['faculty_type'] == faculty]
    ax.plot(subset['Year'], subset['Percentage'], label=faculty, marker='o', color=palette[idx])

# Customizing the plot
ax.set_title("Trends in Employment Among Instructional Staff (1975-2011)", fontsize=16, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Employment Percentage (%)", fontsize=12)
ax.legend(title="Faculty Type", title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='11')

# Improve readability with additional grid lines and ticks
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 34 with threat_id: thread_SktjamXIL304n3AQjMJrnd14
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape data to long format
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' to an integer
data_long['Year'] = data_long['Year'].astype(int)

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 34 with threat_id: thread_SktjamXIL304n3AQjMJrnd14
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape data to long format
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' to an integer
data_long['Year'] = data_long['Year'].astype(int)

# Create a line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o')

# Customize the plot
plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 34 with threat_id: thread_SktjamXIL304n3AQjMJrnd14
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape data to long format
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' to an integer
data_long['Year'] = data_long['Year'].astype(int)

# Set up the figure
plt.figure(figsize=(14, 8))

# Create a line plot
sns.set(style='whitegrid') # Set the aesthetic style of the plots
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, weight='bold', pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Employment', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(visible=True, which='major', color='grey', linestyle='--', linewidth=0.5)

# Highlight a specific time period (e.g., 1990-2000)
plt.axvspan(1990, 2000, color='yellow', alpha=0.1)

# Tight layout to make space for legend
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 37 with threat_id: thread_JpTfgSt2ZnvwX9jjoZ0AXjvZ
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_melted = data.melt(id_vars=["faculty_type"], var_name="year", value_name="percentage")

# Convert the year to a numerical type
data_melted['year'] = data_melted['year'].astype(int)

# Plotting
plt.figure(figsize=(14, 8))
for faculty_type in data['faculty_type'].unique():
    subset = data_melted[data_melted['faculty_type'] == faculty_type]
    plt.plot(subset['year'], subset['percentage'], marker='o', linestyle='', markersize=8, label=faculty_type)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(data_melted['year'].unique()) # Set tick marks to years present in data
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 37 with threat_id: thread_JpTfgSt2ZnvwX9jjoZ0AXjvZ
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_melted = data.melt(id_vars=["faculty_type"], var_name="year", value_name="percentage")

# Convert the year to a numerical type
data_melted['year'] = data_melted['year'].astype(int)

# Plotting
plt.figure(figsize=(14, 8))
for faculty_type in data['faculty_type'].unique():
    subset = data_melted[data_melted['faculty_type'] == faculty_type]
    plt.plot(subset['year'], subset['percentage'], marker='o', label=faculty_type)

plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(data_melted['year'].unique()) # Set tick marks to years present in data
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 37 with threat_id: thread_JpTfgSt2ZnvwX9jjoZ0AXjvZ
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for easier plotting
data_melted = data.melt(id_vars=["faculty_type"], var_name="year", value_name="percentage")

# Convert the year to a numerical type
data_melted['year'] = data_melted['year'].astype(int)

# Plotting
plt.figure(figsize=(14, 8))
for faculty_type in data['faculty_type'].unique():
    subset = data_melted[data_melted['faculty_type'] == faculty_type]
    plt.plot(subset['year'], subset['percentage'], marker='o', label=faculty_type)

# Title and labels
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)

# Legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)

# Annotations and Grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(data_melted['year'].unique(), fontsize=12)  # Set tick marks to years present in data
plt.yticks(fontsize=12)

# Make layout tight
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 39 with threat_id: thread_QnFzdE9Ev0Dewqru5aNf0fjJ
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the dataframe for plotting
data_melted = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Plotting
plt.figure(figsize=(12, 8))
for faculty in data['faculty_type']:
    subset = data_melted[data_melted['faculty_type'] == faculty]
    plt.plot(subset['Year'], subset['Percentage'], 'o-', label=faculty, markersize=5)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 39 with threat_id: thread_QnFzdE9Ev0Dewqru5aNf0fjJ
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the dataframe for plotting
data_melted = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Create a bar plot
plt.figure(figsize=(14, 8))
faculty_types = data['faculty_type'].unique()

# Plot each faculty type with a different color
for i, faculty in enumerate(faculty_types):
    subset = data_melted[data_melted['faculty_type'] == faculty]
    plt.bar(subset['Year'], subset['Percentage'], label=faculty)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 39 with threat_id: thread_QnFzdE9Ev0Dewqru5aNf0fjJ
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the dataframe for plotting
data_melted = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Set figure size
plt.figure(figsize=(14, 8))

# Attributes for plotting
faculty_types = data['faculty_type'].unique()
bar_width = 0.15
years = data.columns[1:]  # Extract years from the columns
indices = range(len(years))

# Plot each faculty type as a separate set of bars
for i, faculty in enumerate(faculty_types):
    subset = data.loc[data['faculty_type'] == faculty, years].values.flatten()
    plt.bar([index + i * bar_width for index in indices], subset, bar_width, label=faculty)

# Customizing plot appearance
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Employment (%)', fontsize=12)
plt.xticks([index + bar_width * (len(faculty_types) - 1) / 2 for index in indices], years, rotation=45)
plt.yticks(range(0, 101, 10))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Faculty Type', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

# Optimizing spacing
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right side for the legend

# Display plot
plt.show()
##################################################
#Question 14.0, Round 40 with threat_id: thread_qiY1mGsTJATKpfmD5eF36Zv7
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert 'year' column to numeric for proper plotting
data_long['year'] = pd.to_numeric(data_long['year'])

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x='year', y='percentage', hue='faculty_type', s=100, palette='Set1')

# Add title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')

# Display the legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 40 with threat_id: thread_qiY1mGsTJATKpfmD5eF36Zv7
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert 'year' column to numeric for proper plotting
data_long['year'] = pd.to_numeric(data_long['year'])

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_long, x='year', y='percentage', hue='faculty_type', marker='o', palette='Set1')

# Add title and labels
plt.title('Instructional Staff Employment Trends (Line Plot)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')

# Display the legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Enhance the layout
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 40 with threat_id: thread_qiY1mGsTJATKpfmD5eF36Zv7
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert 'year' column to numeric for proper plotting
data_long['year'] = pd.to_numeric(data_long['year'])

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create an improved line plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_long, x='year', y='percentage', hue='faculty_type', marker='o', palette='Set1')

# Add an improved title and axis labels with larger font sizes
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)

# Customize ticks for better readability
plt.xticks(data_long['year'].unique(), rotation=45)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)

# Enhance the legend
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')

# Enhance the layout
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 41 with threat_id: thread_kEgfN9wtFmm4AIRtf6qkbpS3
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/path/to/your/file.csv')

# Melt the data to have a long format
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a dot plot
fig, ax = plt.subplots(figsize=(12, 8))
for key, grp in melted_data.groupby(['faculty_type']):
    ax.plot(grp['Year'], grp['Percentage'], marker='o', linestyle='', label=key)

# Customizing the plot
ax.set_xlabel('Year')
ax.set_ylabel('Employment Percentage')
ax.set_title('Instructional Staff Employment Trends Over Time')
ax.legend(title='Faculty Type')
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.show()
##################################################
#Question 14.1, Round 41 with threat_id: thread_kEgfN9wtFmm4AIRtf6qkbpS3
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/path/to/your/file.csv')

# Melt the data to have a long format
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to a numeric type for proper plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Create a line plot with markers
plt.figure(figsize=(12, 8))
for key, grp in melted_data.groupby('faculty_type'):
    plt.plot(grp['Year'], grp['Percentage'], marker='o', linestyle='-', label=key)

# Customizing the plot
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.title('Instructional Staff Employment Trends Over Time')
plt.legend(title='Faculty Type')
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.show()
##################################################
#Question 14.2, Round 41 with threat_id: thread_kEgfN9wtFmm4AIRtf6qkbpS3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/path/to/your/file.csv')

# Melt the data to have a long format
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to a numeric type for proper plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Set a Seaborn style for the plot
sns.set(style='whitegrid')

# Create a line plot with markers
plt.figure(figsize=(14, 8))
sns.lineplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2)

# Customizing the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Faculty Type (%)', fontsize=14)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11')
plt.xticks(rotation=45)
plt.grid(visible=True, linestyle='--', linewidth=0.5)

# Adding an annotation for highlighting a specific trend
plt.annotate('Noticeable Shift', xy=(1999, 21.8), xytext=(2003, 30),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='darkred')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 42 with threat_id: thread_xBM5nQ5rK30gWbOXHeRVHXS2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format
long_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert year to integer for plotting
long_data['Year'] = long_data['Year'].astype(int)

# Create dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', style='faculty_type')

# Improve plot aesthetics
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.1, Round 42 with threat_id: thread_xBM5nQ5rK30gWbOXHeRVHXS2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format
long_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert year to integer for plotting
long_data['Year'] = long_data['Year'].astype(int)

# Set the style for the plot
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', marker='o')

# Improve plot aesthetics
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.2, Round 42 with threat_id: thread_xBM5nQ5rK30gWbOXHeRVHXS2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to a long format
long_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert year to integer for plotting
long_data['Year'] = long_data['Year'].astype(int)

# Set the style for the plot
sns.set(style="whitegrid", palette="Set1")

# Create a line plot
plt.figure(figsize=(14, 8))
line_plot = sns.lineplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Improve plot aesthetics
plt.title('Trends in Instructional Staff Employment by Faculty Type (1975-2011)', fontsize=18, weight='bold')
plt.xlabel('Year', fontsize=14, labelpad=10)
plt.ylabel('Percentage of Total Faculty (%)', fontsize=14, labelpad=10)
plt.xticks(rotation=45)
plt.ylim(0, 50)  # Set a limit to the y-axis for better visibility and comparison
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Add annotations for clarity
for i, row in long_data.iterrows():
    if row['Year'] == 2011:  # Annotate only the last year for each line
        plt.text(row['Year'] + 0.2, row['Percentage'], row['faculty_type'], 
                 horizontalalignment='left', size='small', color='black', weight='semibold')

# Show plot
plt.show()
##################################################
#Question 14.0, Round 43 with threat_id: thread_tmy3mGGxeBXiNXwUF9v1bxmx
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Convert data to long format for plotting
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a dot plot
plt.figure(figsize=(12, 8))
for faculty in data['faculty_type'].unique():
    subset = data_long[data_long['faculty_type'] == faculty]
    plt.plot(subset['Year'], subset['Percentage'], 'o-', label=faculty)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 43 with threat_id: thread_tmy3mGGxeBXiNXwUF9v1bxmx
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (assumed to be already loaded as in previous snippets)
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Convert data to long format for plotting
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a line plot
plt.figure(figsize=(12, 8))
for faculty in data['faculty_type'].unique():
    subset = data_long[data_long['faculty_type'] == faculty]
    plt.plot(subset['Year'], subset['Percentage'], marker='o', label=faculty)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 43 with threat_id: thread_tmy3mGGxeBXiNXwUF9v1bxmx
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Convert data to long format for plotting
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a line plot
plt.figure(figsize=(14, 8))
for faculty in data['faculty_type'].unique():
    subset = data_long[data_long['faculty_type'] == faculty]
    plt.plot(subset['Year'], subset['Percentage'], marker='o', label=faculty)

# Improving the plot aesthetics and text
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Faculty (%)', fontsize=14)

plt.xticks(rotation=45)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Enhancing the plot with grid and space
plt.tight_layout(pad=2.0)

plt.show()
##################################################
#Question 14.0, Round 44 with threat_id: thread_zHnGFDwFWErXbPJQEoqamcUL
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set(style="whitegrid")

# Melt the dataset to long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.stripplot(x='year', y='percentage', hue='faculty_type', data=data_long, size=7, jitter=True, dodge=True)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 44 with threat_id: thread_zHnGFDwFWErXbPJQEoqamcUL
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set(style="whitegrid")

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_long, x='year', y='percentage', hue='faculty_type', marker='o')

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 44 with threat_id: thread_zHnGFDwFWErXbPJQEoqamcUL
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set(style="whitegrid")

# Define the color palette
palette = sns.color_palette("Set2", n_colors=len(data['faculty_type'].unique()))

# Plot using seaborn
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_long, x='year', y='percentage', hue='faculty_type', marker='o', palette=palette)

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', title_fontsize='13', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='11')
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 45 with threat_id: thread_yL0T4PaZO0KBP1p8h6NKmQP5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV-like content
file_path = '/path/to/your/file.csv'  # Update this with the correct path
data = pd.read_csv(file_path)

# Set the faculty type as the index for easier plotting
data.set_index('faculty_type', inplace=True)

# Transpose the data to have years on the x-axis
data = data.transpose()

# Plotting the dot plot
plt.figure(figsize=(10, 6))

# Plot each faculty type as a different colored line with markers
for column in data.columns:
    plt.plot(data.index, data[column], marker='o', label=column)

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('Instructional Staff Employment Trends Over Time')
plt.legend(title='Faculty Type')
plt.grid(True)

# Display the plot
plt.show()
##################################################
#Question 14.1, Round 45 with threat_id: thread_yL0T4PaZO0KBP1p8h6NKmQP5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV-like content
# Assuming the path to your local file is correct
file_path = '/path/to/your/file.csv'  # Update this with the correct path
data = pd.read_csv(file_path)

# Set the faculty type as the index for easier plotting
data.set_index('faculty_type', inplace=True)

# Transpose the data to have years on the x-axis
data = data.transpose()

# Plotting a stacked bar plot
plt.figure(figsize=(12, 8))

# Plotting each faculty type on the bar plot
bottom = pd.Series([0] * data.shape[0], index=data.index)
for column in data.columns:
    plt.bar(data.index, data[column], bottom=bottom, label=column)
    bottom += data[column]

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('Instructional Staff Employment Trends Over Time (Stacked Barchart)')
plt.legend(title='Faculty Type')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 45 with threat_id: thread_yL0T4PaZO0KBP1p8h6NKmQP5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV-like content
# Assuming the path to your local file is correct
file_path = '/path/to/your/file.csv'  # Update this with the correct path
data = pd.read_csv(file_path)

# Set the faculty type as the index for easier plotting
data.set_index('faculty_type', inplace=True)

# Transpose the data to have years on the x-axis
data = data.transpose()

# Plotting a stacked bar plot
plt.figure(figsize=(14, 8))

# Colors for each faculty type for a clearer distinction
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700']

# Plotting each faculty type on the bar plot
bottom = pd.Series([0] * data.shape[0], index=data.index)
for i, column in enumerate(data.columns):
    plt.bar(data.index, data[column], bottom=bottom, label=column, color=colors[i % len(colors)])
    bottom += data[column]

# Enhance labels and title for clarity
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Staff (%)', fontsize=12)
plt.title('Instructional Staff Employment Trends Over Time', fontsize=14, fontweight='bold')

# Improve legend styling
plt.legend(title='Faculty Type', title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

# Adding grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 46 with threat_id: thread_fPXGQvw4dXcd72ViV5MpbaQQ
import pandas as pd

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the plot size
plt.figure(figsize=(10, 6))

# Transpose the data so that years are along the index
data_transposed = data.set_index('faculty_type').T

# Plot each faculty type
for faculty_type in data_transposed.columns:
    plt.plot(data_transposed.index, data_transposed[faculty_type], 'o-', label=faculty_type)

# Add title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 46 with threat_id: thread_fPXGQvw4dXcd72ViV5MpbaQQ
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the plot size
plt.figure(figsize=(10, 6))

# Transpose the data so that years are along the index
data_transposed = data.set_index('faculty_type').T

# Plot the stacked area plot
plt.stackplot(data_transposed.index, data_transposed.T, labels=data_transposed.columns)

# Add title and labels
plt.title('Instructional Staff Employment Trends (Stacked Area Plot)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 46 with threat_id: thread_fPXGQvw4dXcd72ViV5MpbaQQ
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the plot size
plt.figure(figsize=(12, 8))

# Transpose the data so that years are along the index
data_transposed = data.set_index('faculty_type').T

# Plot the stacked area plot
plt.stackplot(data_transposed.index, data_transposed.T, labels=data_transposed.columns, alpha=0.8)

# Add title and labels with improvements
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Staff', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Improve legend with custom positioning and font size
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=12)
plt.grid(True, axis='y', linestyle='--', linewidth=0.7)

# Add a horizontal line at certain percentage (e.g., 50%) to provide context
plt.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, label='50% Line')

# Add annotations or text for additional context if necessary
# Example: plt.text(x='2000', y=70, s='Note: Data up to 2011', fontsize=10, color='red')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 47 with threat_id: thread_QyH42TjgKA8iE0uP4mSQsUtY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '/path/to/your/file.csv'  # Update this to the path where your file is located
data = pd.read_csv(file_path)

# Transform data from wide to long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='employment')

# Convert year to integer
data_long['year'] = data_long['year'].astype(int)

# Plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create a dot plot
sns.scatterplot(data=data_long, x='year', y='employment', hue='faculty_type', style='faculty_type', s=100)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 47 with threat_id: thread_QyH42TjgKA8iE0uP4mSQsUtY
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/path/to/your/file.csv'  # Update this to the path where your file is located
data = pd.read_csv(file_path)

# Transform data from wide to long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='employment')

# Convert year to integer
data_long['year'] = data_long['year'].astype(int)

# Plot
plt.figure(figsize=(12, 8))

# Create a line plot
for faculty_type in data['faculty_type']:
    subset = data_long[data_long['faculty_type'] == faculty_type]
    plt.plot(subset['year'], subset['employment'], marker='o', label=faculty_type)

plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 47 with threat_id: thread_QyH42TjgKA8iE0uP4mSQsUtY
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/path/to/your/file.csv'  # Update this to the path where your file is located
data = pd.read_csv(file_path)

# Transform data from wide to long format for easier plotting
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='employment')

# Convert year to integer
data_long['year'] = data_long['year'].astype(int)

# Plot
plt.figure(figsize=(14, 8))

# Create a line plot with more distinct colors
colors = plt.cm.viridis(range(len(data['faculty_type'])))
for i, faculty_type in enumerate(data['faculty_type']):
    subset = data_long[data_long['faculty_type'] == faculty_type]
    plt.plot(subset['year'], subset['employment'], marker='o', label=faculty_type, color=colors[i])

# Enhancements for labels, title, and legend
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 48 with threat_id: thread_2lEfYqBnSAnH0gvn75AtlX0X
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Reshape the data for plotting
df_melted = df.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the dot plot
plt.figure(figsize=(12, 8))
for key, grp in df_melted.groupby(['faculty_type']):
    plt.plot(grp['Year'], grp['Percentage'], marker='o', linestyle='', label=key)

# Add plot details
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.1, Round 48 with threat_id: thread_2lEfYqBnSAnH0gvn75AtlX0X
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Reshape the data for plotting
df_melted = df.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the line plot
plt.figure(figsize=(12, 8))
for key, grp in df_melted.groupby(['faculty_type']):
    plt.plot(grp['Year'], grp['Percentage'], marker='o', linestyle='-', label=key)

# Add plot details
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.2, Round 48 with threat_id: thread_2lEfYqBnSAnH0gvn75AtlX0X
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Reshape the data for plotting
df_melted = df.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the line plot
plt.figure(figsize=(14, 8))
for key, grp in df_melted.groupby(['faculty_type']):
    plt.plot(grp['Year'], grp['Percentage'], marker='o', linestyle='-', label=key)

# Add plot details with improved labels
plt.title('Trends in Instructional Staff Employment \n (1975 - 2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Faculty', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='12')
plt.grid(visible=True, linestyle='--', linewidth=0.5)

# Enhance layout
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 50 with threat_id: thread_jvEUlP4YfyIz1n3zl5Llo7aZ
# Importing necessary libraries
import pandas as pd

# Reading the uploaded file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Displaying the first few rows of the data to understand its structure
data.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Transforming the data from wide to long format
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Setting up the plot size
plt.figure(figsize=(12, 8))

# Creating the dot plot using seaborn
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Adding title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Showing the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 50 with threat_id: thread_jvEUlP4YfyIz1n3zl5Llo7aZ
import matplotlib.pyplot as plt
import seaborn as sns

# Transforming the data from wide to long format
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Converting 'Year' to integer type for proper plotting
data_long['Year'] = data_long['Year'].astype(int)

# Setting up the plot size
plt.figure(figsize=(12, 8))

# Creating the line plot using seaborn
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Adding title and labels
plt.title('Instructional Staff Employment Trends (Line Plot)')
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Showing the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 50 with threat_id: thread_jvEUlP4YfyIz1n3zl5Llo7aZ
import matplotlib.pyplot as plt
import seaborn as sns

# Transforming the data from wide to long format
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Converting 'Year' to integer type for proper plotting
data_long['Year'] = data_long['Year'].astype(int)

# Setting up the plot size and style
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Creating the line plot using seaborn
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Adding title and labels with more informative descriptions
plt.title('Trends in Instructional Staff Employment by Faculty Type (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)

# Enhancing the X-axis ticks for better readability
plt.xticks(rotation=45)
plt.xticks(data_long['Year'].unique())  # Show only available years on the X-axis

# Moving the legend outside the plot for clarity with improved description
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize='13')

# Show grid for better readability of the percentages
plt.grid(visible=True)

# Adjust layout to reduce clipping of tick-labels or legend
plt.tight_layout()

# Showing the plot
plt.show()
##################################################
#Question 14.0, Round 51 with threat_id: thread_NX8Qhld586i44O7d48iI2R27
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years in one column and the values in another
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Set the theme for the plot
sns.set_theme(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='Percentage', y='faculty_type', hue='Year', palette="deep", s=100)

# Customize plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Percentage of Total Employment')
plt.ylabel('Faculty Type')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.1, Round 51 with threat_id: thread_NX8Qhld586i44O7d48iI2R27
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years in one column and the values in another
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to integer for correct plotting
data_melted['Year'] = data_melted['Year'].astype(int)

# Set the theme for the plot
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', marker='o', palette="deep")

# Customize plot
plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Employment')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(data_melted['Year'].unique())  # Ensure all years are shown on x-axis
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.2, Round 51 with threat_id: thread_NX8Qhld586i44O7d48iI2R27
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years in one column and the values in another
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to integer for correct plotting
data_melted['Year'] = data_melted['Year'].astype(int)

# Set the theme for the plot
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', marker='o', palette="tab10", linewidth=2.5, markersize=8)

# Customize plot
plt.title('Trends in Instructional Staff Employment at Educational Institutions (1975-2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Employment (%)', fontsize=14)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(data_melted['Year'].unique(), rotation=45)  # Ensure all years are shown on x-axis
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.grid(True)

# Optimize layout
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 52 with threat_id: thread_94EUZVKwGnBmwMVWjblzwS20
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'your_file_path_here.csv'
data = pd.read_csv(file_path)

# Reshape the data for plotting
data_long = data.melt(id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Convert year to numeric for better plotting
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Create a dot plot
plt.figure(figsize=(14, 8))
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Improve plot aesthetics
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.xticks(data_long['Year'].unique())
plt.legend(title='Faculty Type')
plt.grid(True)

# Show plot
plt.show()
##################################################
#Question 14.1, Round 52 with threat_id: thread_94EUZVKwGnBmwMVWjblzwS20
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data for plotting
data_long = data.melt(id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Convert year to numeric for better plotting
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Create a line plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Improve plot aesthetics
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.xticks(data_long['Year'].unique())
plt.legend(title='Faculty Type')
plt.grid(True)

# Show plot
plt.show()
##################################################
#Question 14.2, Round 52 with threat_id: thread_94EUZVKwGnBmwMVWjblzwS20
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data for plotting
data_long = data.melt(id_vars=['faculty_type'], var_name='Year', value_name='Percentage')

# Convert year to numeric for better plotting
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Set the style of the plot
sns.set_style("whitegrid")

# Create a line plot
plt.figure(figsize=(14, 8))
line_plot = sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5)

# Improve plot aesthetics
plt.title("Trends in Instructional Staff Employment (1975 - 2011)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Employment Percentage (%)", fontsize=14)
plt.xticks(data_long['Year'].unique(), fontsize=12)
plt.yticks(fontsize=12)

# Customize legend
plt.legend(title='Faculty Type', title_fontsize=13, fontsize=11, loc='upper right')

# Grid settings
plt.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# Tight layout for better appearance
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 54 with threat_id: thread_9tCZLL2W9CcMawPjA3i4TmK5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Prepare the data for plotting
years = data.columns[1:]  # The years are the column names excluding the first column
faculty_types = data['faculty_type']

# Plotting
plt.figure(figsize=(10, 6))

# Dot plot
for index, row in data.iterrows():
    plt.plot(years, row[1:], 'o-', label=row['faculty_type'])

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.1, Round 54 with threat_id: thread_9tCZLL2W9CcMawPjA3i4TmK5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Prepare the data for plotting
years = data.columns[1:].astype(int)  # Convert year columns to integers
employment_data = data.iloc[:, 1:]

# Plotting
plt.figure(figsize=(10, 6))

# Stacked area plot
plt.stackplot(years, employment_data.T, labels=data['faculty_type'], alpha=0.7)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.2, Round 54 with threat_id: thread_9tCZLL2W9CcMawPjA3i4TmK5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Prepare the data for plotting
years = data.columns[1:].astype(int)  # Convert year columns to integers
employment_data = data.iloc[:, 1:]

# Set a color palette
palette = sns.color_palette("Set2", len(data['faculty_type']))

# Plotting
plt.figure(figsize=(12, 7))

# Stacked area plot with improved labels
plt.stackplot(years, employment_data.T, labels=data['faculty_type'], colors=palette, alpha=0.8)

# Improved title and labels
plt.title('Trends in Instructional Staff Employment Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage (%)', fontsize=12)
plt.xticks(years, rotation=45)
plt.yticks(range(0, 101, 10))

# Improved legend
plt.legend(loc='upper right', title='Faculty Type')

# Adding gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 55 with threat_id: thread_gVgiJcJnamz6b3mNekohdqeb
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Transpose the data to have years as rows and faculty types as columns
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert the 'year' column to integer for plotting
data_melted['year'] = data_melted['year'].astype(int)

# Plot the data
for key, grp in data_melted.groupby(['faculty_type']):
    plt.plot(grp['year'], grp['percentage'], marker='o', linestyle='', label=key)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 55 with threat_id: thread_gVgiJcJnamz6b3mNekohdqeb
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Transpose and prepare the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')
data_melted['year'] = data_melted['year'].astype(int)

# Plot the data with lines and markers
for key, grp in data_melted.groupby(['faculty_type']):
    plt.plot(grp['year'], grp['percentage'], marker='o', linestyle='-', label=key)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends - Line Plot')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 55 with threat_id: thread_gVgiJcJnamz6b3mNekohdqeb
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the figure and axis with improved size and dpi
plt.figure(figsize=(14, 8), dpi=100)

# Transpose and prepare the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')
data_melted['year'] = data_melted['year'].astype(int)

# Plot the data with lines and markers
for key, grp in data_melted.groupby(['faculty_type']):
    plt.plot(grp['year'], grp['percentage'], marker='o', linestyle='-', label=key)

# Enhancing labels and title
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.title('Trends in Instructional Staff Employment Over Time', fontsize=16)

# Improve legend
plt.legend(title='Faculty Type', fontsize=12, title_fontsize=14, loc='upper left')

# Add gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Add x and y axis major and minor grids
plt.grid(which='minor', linestyle=':', linewidth='0.5')

# Improve ticks
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 56 with threat_id: thread_qVnzf9JBBvMmzpDZeZ7NZaaV
import matplotlib.pyplot as plt

# Transpose the data to have years as rows
data_long = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Plotting
plt.figure(figsize=(10, 6))

# Plot each faculty type as a separate line with dots
for faculty in data['faculty_type']:
    faculty_data = data_long[data_long['faculty_type'] == faculty]
    plt.plot(faculty_data['Year'], faculty_data['Percentage'], marker='o', label=faculty)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(visible=True)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 56 with threat_id: thread_qVnzf9JBBvMmzpDZeZ7NZaaV
import matplotlib.pyplot as plt

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Stack area plot
for i, faculty in enumerate(data['faculty_type']):
    plt.fill_between(data.columns[1:], data.iloc[i].values[1:], 
                     data.iloc[i-1].values[1:] if i != 0 else 0,
                     label=faculty)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', loc='upper left')
plt.grid(visible=True)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 56 with threat_id: thread_qVnzf9JBBvMmzpDZeZ7NZaaV
import matplotlib.pyplot as plt

# Colors for each faculty type
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Stack area plot with enhanced labeling and colors
cumulative_data = pd.DataFrame()
for i, faculty in enumerate(data['faculty_type']):
    cumulative_data[faculty] = data.iloc[i].values[1:]
    plt.fill_between(data.columns[1:], cumulative_data[faculty] + (cumulative_data.sum(axis=1).shift(1).fillna(0) if i != 0 else 0),
                     cumulative_data.sum(axis=1) if i != 0 else 0,
                     label=faculty, color=colors[i])

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Employment (%)', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Faculty Type', loc='upper left', fontsize=10, title_fontsize=12)
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 57 with threat_id: thread_Vq12HOq4PykdplpiulKXmnmK
import pandas as pd

# Load the data from the uploaded file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()


import matplotlib.pyplot as plt

# Transpose the data to have years as rows
data_long = data.set_index('faculty_type').T

# Plotting
plt.figure(figsize=(10, 6))
for faculty_type in data_long.columns:
    plt.plot(data_long.index, data_long[faculty_type], marker='o', label=faculty_type)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 57 with threat_id: thread_Vq12HOq4PykdplpiulKXmnmK
# Plotting
plt.figure(figsize=(10, 6))

# Create an area plot
plt.stackplot(data_long.index, *[data_long[faculty_type] for faculty_type in data_long.columns], labels=data_long.columns)

# Customize the plot
plt.title('Instructional Staff Employment Trends - Area Plot')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', loc='upper right')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 57 with threat_id: thread_Vq12HOq4PykdplpiulKXmnmK
# Plotting
plt.figure(figsize=(12, 8))

# Define colors for each faculty type
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

# Create an area plot
plt.stackplot(data_long.index, 
              *[data_long[faculty_type] for faculty_type in data_long.columns], 
              labels=data_long.columns, 
              colors=colors)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Staff', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(title='Faculty Type', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, title_fontsize=12)
plt.grid(linestyle='--', alpha=0.7)
plt.margins(0, 0)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 58 with threat_id: thread_PBxUdgJF10fMG7HDS9VGukgW
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to a numeric type
data_melted['Year'] = data_melted['Year'].astype(int)

# Plotting
plt.figure(figsize=(12, 8))

# Create a dot plot
for faculty_type in data['faculty_type']:
    plt.plot('Year', 'Percentage', data=data_melted[data_melted['faculty_type'] == faculty_type], 
             marker='o', linestyle='', label=faculty_type)

# Adding titles and labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(data_melted['Year'].unique())  # Ensuring all years are shown

# Adding legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 58 with threat_id: thread_PBxUdgJF10fMG7HDS9VGukgW
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to a numeric type
data_melted['Year'] = data_melted['Year'].astype(int)

# Plotting
plt.figure(figsize=(12, 8))

# Create a line plot with markers
for faculty_type in data['faculty_type']:
    plt.plot('Year', 'Percentage', data=data_melted[data_melted['faculty_type'] == faculty_type], 
             marker='o', label=faculty_type)

# Adding titles and labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(data_melted['Year'].unique())  # Ensuring all years are shown

# Adding legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.grid(True)  # Adding grid lines for better readability
plt.show()
##################################################
#Question 14.2, Round 58 with threat_id: thread_PBxUdgJF10fMG7HDS9VGukgW
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transpose the data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert Year to a numeric type
data_melted['Year'] = data_melted['Year'].astype(int)

# Plotting
plt.figure(figsize=(14, 8))

# Create a line plot with markers
for faculty_type in data['faculty_type']:
    plt.plot('Year', 'Percentage', data=data_melted[data_melted['faculty_type'] == faculty_type], 
             marker='o', label=faculty_type)

# Adding titles and labels
plt.title('Trends in Instructional Staff Employment by Faculty Type (1975-2011)', fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Employment Percentage (%)', fontsize=14, fontweight='bold')
plt.xticks(data_melted['Year'].unique(), fontsize=12)
plt.yticks(fontsize=12)

# Adding legend
plt.legend(title='Faculty Type', title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='11')

# Adding grid for better readability
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Tight layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 59 with threat_id: thread_bMmuPVjkbd0uqfasaGCeszBD
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe
melted_data = data.melt(id_vars="faculty_type", var_name="year", value_name="proportion")

# Create the dot plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=melted_data, x='year', y='faculty_type', size='proportion', sizes=(20, 200), legend=False)

# Add title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Faculty Type')

# Rotate the x-axis labels for readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 59 with threat_id: thread_bMmuPVjkbd0uqfasaGCeszBD
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the index to faculty type for easier plotting
data.set_index('faculty_type', inplace=True)

# Transpose the data to have years as the index
transposed_data = data.T

# Create the line plot
plt.figure(figsize=(12, 8))
for faculty in transposed_data.columns:
    plt.plot(transposed_data.index, transposed_data[faculty], marker='o', label=faculty)

# Add title and labels
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')

# Add legend
plt.legend(loc='upper right')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.grid(True)
plt.show()
##################################################
#Question 14.2, Round 59 with threat_id: thread_bMmuPVjkbd0uqfasaGCeszBD
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set a visual style
sns.set_style("whitegrid")

# Set the index to faculty type for easier plotting
data.set_index('faculty_type', inplace=True)

# Transpose the data to have years as the index
transposed_data = data.T

# Create the line plot
plt.figure(figsize=(14, 8))
for faculty in transposed_data.columns:
    plt.plot(transposed_data.index, transposed_data[faculty], marker='o', label=faculty)

# Add a more descriptive title
plt.title('Employment Trends of Instructional Staff from 1975 to 2011', fontsize=16, fontweight='bold')

# Add labels with units to the axes
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Proportion of Total Faculty (%)', fontsize=12, fontweight='bold')

# Add legend with a descriptive title
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', loc='upper right')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid lines
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')

# Improve spacing with tight layout
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 60 with threat_id: thread_zWy9W9WCK9F8seOfnoGrjhD2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to convert it into a long format suitable for seaborn
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' column to a numeric type for better plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Set the style of the plot
sns.set_style('whitegrid')

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.1, Round 60 with threat_id: thread_zWy9W9WCK9F8seOfnoGrjhD2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to convert it into a long format suitable for seaborn
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' column to a numeric type for better plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Set the style of the plot
sns.set_style('whitegrid')

# Create the line plot with markers
plt.figure(figsize=(12, 8))
sns.lineplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', marker='o', dashes=False)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 60 with threat_id: thread_zWy9W9WCK9F8seOfnoGrjhD2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to convert it into a long format suitable for seaborn
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' column to a numeric type for better plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Set the style of the plot
sns.set_style('whitegrid')

# Create the line plot with markers
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=melted_data, 
    x='Year', 
    y='Percentage', 
    hue='faculty_type', 
    marker='o', 
    dashes=False,
    linewidth=2.5
)

# Improve the title, axis labels, and legend
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=14, labelpad=10)
plt.ylabel('Percentage of Employment (%)', fontsize=14, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Enhance legend
plt.legend(
    title='Faculty Type', 
    title_fontsize='13', 
    fontsize='11', 
    loc='upper left', 
    bbox_to_anchor=(1.05, 1)
)

# Add gridlines for better readability
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5)

# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 61 with threat_id: thread_kzeBUc1SIlZy4D3yVydN8K2p
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Preprocess Data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to integer
data_melted['Year'] = data_melted['Year'].astype(int)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Title and labels
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')

# Show legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 61 with threat_id: thread_kzeBUc1SIlZy4D3yVydN8K2p
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Preprocess Data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to integer
data_melted['Year'] = data_melted['Year'].astype(int)

# Plotting
plt.figure(figsize=(12, 7))
sns.lineplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', marker='o')

# Title and labels
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage')

# Show legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 61 with threat_id: thread_kzeBUc1SIlZy4D3yVydN8K2p
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Preprocess Data for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to integer
data_melted['Year'] = data_melted['Year'].astype(int)

# Set plot style
sns.set(style='whitegrid')

# Plotting
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_melted, x='Year', y='Percentage', hue='faculty_type', marker='o', linewidth=2.5, style='faculty_type', dashes=False)

# Title and labels
plt.title('Trends in Employment of Instructional Staff (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Employment', fontsize=12)

# Improve legend
plt.legend(title='Type of Faculty', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize='13')

# Customize ticks
plt.xticks(range(1975, 2012, 4), rotation=45)
plt.yticks(fontsize=10)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot with tight layout
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 62 with threat_id: thread_sNk1IN1vr95pDIlPWmg9Msrg
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Set the 'faculty_type' as the index
data.set_index('faculty_type', inplace=True)

# Transpose the data to prepare it for plotting
data_T = data.T

# Create a dot plot
plt.figure(figsize=(10, 6))
for column in data_T.columns:
    plt.plot(data_T.index, data_T[column], 'o-', label=column)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 62 with threat_id: thread_sNk1IN1vr95pDIlPWmg9Msrg
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Set the 'faculty_type' as the index
data.set_index('faculty_type', inplace=True)

# Transpose the data to prepare it for plotting
data_T = data.T

# Create a stacked area plot
plt.figure(figsize=(12, 7))
data_T.plot(kind='area', stacked=True, figsize=(12, 7), colormap='Paired', alpha=0.8)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends (Stacked Area Plot)')

# Move the legend outside the plot
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 62 with threat_id: thread_sNk1IN1vr95pDIlPWmg9Msrg
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/file-5riwCDAHXme7U6mfNZwCLe')

# Set the 'faculty_type' as the index
data.set_index('faculty_type', inplace=True)

# Transpose the data to prepare it for plotting
data_T = data.T

# Create a stacked area plot
plt.figure(figsize=(12, 8))
data_T.plot(kind='area', stacked=True, figsize=(12, 8), colormap='Accent', alpha=0.85)

# Customize the plot
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Faculty (%)', fontsize=12)
plt.title('Trends in Instructional Staff Employment Over Time (1975-2011)', fontsize=14)

# Improve legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)

# Add grid lines for better readability
plt.grid(color='grey', linestyle='--', linewidth=0.5)

# Display the plot with tight layout
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend
plt.show()
##################################################
#Question 14.0, Round 63 with threat_id: thread_zVnPCQEQsSOMQWFyjFuqcK5Z
import matplotlib.pyplot as plt

# Create a dot plot
fig, ax = plt.subplots(figsize=(12, 8))

# Years (columns of data except for the first column)
years = data.columns[1:]

# Plotting each faculty type
for index, row in data.iterrows():
    ax.plot(years, row[1:], marker='o', label=row['faculty_type'], linestyle='')

# Customize the plot
ax.set_xticks(years)
ax.set_xticklabels(years)
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Faculty Type')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 63 with threat_id: thread_zVnPCQEQsSOMQWFyjFuqcK5Z
import matplotlib.pyplot as plt

# Create a line plot
fig, ax = plt.subplots(figsize=(12, 8))

# Years (columns of data except for the first column)
years = data.columns[1:]

# Plotting each faculty type
for index, row in data.iterrows():
    ax.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Customize the plot
ax.set_xticks(years)
ax.set_xticklabels(years)
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Faculty Type')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.grid(True)
plt.show()
##################################################
#Question 14.2, Round 63 with threat_id: thread_zVnPCQEQsSOMQWFyjFuqcK5Z
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for better aesthetics
sns.set_style("whitegrid")

# Create a line plot
fig, ax = plt.subplots(figsize=(14, 8))

# Years (columns of data except for the first column)
years = data.columns[1:]

# Define markers for different lines
markers = ['o', 's', 'D', '^', 'v']

# Define a color palette
colors = sns.color_palette("husl", len(data))

# Plotting each faculty type
for index, (row, marker, color) in enumerate(zip(data.iterrows(), markers, colors)):
    ax.plot(years, row[1][1:], marker=marker, label=row[1]['faculty_type'], color=color, linewidth=2)

# Customize the plot
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Percentage of Faculty Type (%)', fontsize=12)
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14, fontweight='bold')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Improve layout and grid
plt.tight_layout()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 64 with threat_id: thread_HArRzzM7hjx7gUEO06REuJ56
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (assuming the DataFrame 'data' is already available)
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to a numeric type for proper plotting
data_long['Year'] = data_long['Year'].astype(int)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', style='faculty_type', s=100)

# Add titles and labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 64 with threat_id: thread_HArRzzM7hjx7gUEO06REuJ56
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (assuming the DataFrame 'data' is already available)
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to a numeric type for proper plotting
data_long['Year'] = data_long['Year'].astype(int)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o')

# Add titles and labels
plt.title('Instructional Staff Employment Trends', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 64 with threat_id: thread_HArRzzM7hjx7gUEO06REuJ56
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (assuming the DataFrame 'data' is already available)
data_long = pd.melt(data, id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to a numeric type for proper plotting
data_long['Year'] = data_long['Year'].astype(int)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a line plot with improved labels and aesthetics
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_long, x='Year', y='Percentage', hue='faculty_type', marker='o')

# Customize titles and labels
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=14, labelpad=10)
plt.ylabel('Employment Percentage (%)', fontsize=14, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Customize the legend
plt.legend(title='Faculty Type', title_fontsize=13, fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust the plot layout
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 65 with threat_id: thread_1qtAXOw3z6rDHledMwTmYOj9
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data to a long format
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_long, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100, palette='muted')

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 65 with threat_id: thread_1qtAXOw3z6rDHledMwTmYOj9
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data to a long format
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting a line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_long, x='year', y='percentage', hue='faculty_type', marker='o', palette='muted')

plt.title('Instructional Staff Employment Trends (Line Plot)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 65 with threat_id: thread_1qtAXOw3z6rDHledMwTmYOj9
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data to a long format
data_long = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Creating an improved line plot
plt.figure(figsize=(14, 8))

# Define a more distinct color palette
palette = sns.color_palette("husl", len(data['faculty_type'].unique()))

# Plotting with aesthetic improvements
sns.lineplot(data=data_long, x='year', y='percentage', hue='faculty_type', marker='o', palette=palette)

# Updating labels and title
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Faculty (%)', fontsize=14)

# Modifying the legend
plt.legend(title='Faculty Type', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')

# Enhancing readability
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 66 with threat_id: thread_k6a0ck28D3uo5Nmx5jaNS6z5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot each faculty type as a separate line of dots
for index, row in df.iterrows():
    plt.plot(df.columns[1:], row[1:], 'o-', label=row['faculty_type'])

# Adding the labels and title
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.title('Instructional Staff Employment Trends')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 66 with threat_id: thread_k6a0ck28D3uo5Nmx5jaNS6z5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot a stacked area chart
plt.stackplot(df.columns[1:], df.iloc[:, 1:], labels=df['faculty_type'])

# Adding the labels and title
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.title('Instructional Staff Employment Trends (Stacked Area Plot)')
plt.xticks(rotation=45)
plt.legend(loc='upper left', title='Faculty Type')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.2, Round 66 with threat_id: thread_k6a0ck28D3uo5Nmx5jaNS6z5
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Set the figure size
plt.figure(figsize=(14, 9))

# Plot a stacked area chart
plt.stackplot(df.columns[1:], df.iloc[:, 1:], labels=df['faculty_type'], alpha=0.8)

# Adding the labels and title with improved descriptions
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of Total Employment', fontsize=12, fontweight='bold')
plt.title('Trends in Instructional Staff Employment Over Time (1975-2011)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Improved legend with clearer title and placement
plt.legend(loc='upper left', title='Type of Faculty', fontsize=10, title_fontsize='13')
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)

# Aesthetic adjustments
plt.tight_layout()
plt.axhline(0, color='black', linewidth=0.8)

# Show plot
plt.show()
##################################################
#Question 14.0, Round 69 with threat_id: thread_ECVizaY0pKrNBsr1bPtDU5el
import matplotlib.pyplot as plt

# Setting the plot style
plt.style.use('seaborn-darkgrid')

# Creating a dot plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each faculty type
for index, row in data.iterrows():
    ax.plot(data.columns[1:], row[1:], marker='o', label=row['faculty_type'])

# Adding titles and labels
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')

# Adding legend
ax.legend()

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 69 with threat_id: thread_ECVizaY0pKrNBsr1bPtDU5el
import matplotlib.pyplot as plt

# Prepare data for the stacked area plot
years = data.columns[1:].astype(int)
employment_data = data.set_index('faculty_type').transpose()

# Plotting a stacked area chart
fig, ax = plt.subplots(figsize=(12, 8))

ax.stackplot(years, employment_data.T, labels=employment_data.columns, alpha=0.8)

# Adding titles and labels
ax.set_title('Instructional Staff Employment Trends: Stacked Area Plot')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')

# Adding legend
ax.legend(loc='upper right')

# Display the plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 69 with threat_id: thread_ECVizaY0pKrNBsr1bPtDU5el
import matplotlib.pyplot as plt

# Prepare data for the stacked area plot
years = data.columns[1:].astype(int)
employment_data = data.set_index('faculty_type').transpose()

# Plotting a stacked area chart
fig, ax = plt.subplots(figsize=(14, 8))

# Customization of the plot with improved colors and alpha for better visibility
ax.stackplot(years, employment_data.T, labels=employment_data.columns, alpha=0.85, 
             colors=['#FF5733', '#33FF57', '#3357FF', '#F7FF33', '#FF33F7'])

# Adding titles and labels with improved formatting
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold', color='#333333')
ax.set_xlabel('Year', fontsize=12, fontweight='bold', color='#333333')
ax.set_ylabel('Percentage of Total Employment', fontsize=12, fontweight='bold', color='#333333')

# Adjusting the legend and its position
ax.legend(title='Faculty Type', fontsize=10, title_fontsize='11', loc='upper left')

# Enabling the grid for better readability
ax.grid(True, linestyle='--', linewidth=0.5)

# Fine-tuning the ticks for better readability
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Tight layout to avoid overlap
plt.tight_layout()

# Display the improved plot
plt.show()
##################################################
#Question 14.0, Round 70 with threat_id: thread_K8R3P4Re7gW2PepwRg5XI1o5
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe from wide to long format for easier plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Set the plot style
plt.style.use('seaborn-whitegrid')

# Create the dot plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type as a separate series
for faculty in melted_data['faculty_type'].unique():
    subset = melted_data[melted_data['faculty_type'] == faculty]
    ax.plot(subset['Year'], subset['Percentage'], 'o-', label=faculty)

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Employment Percentage (%)')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 70 with threat_id: thread_K8R3P4Re7gW2PepwRg5XI1o5
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe from wide to long format
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Set the plot style
plt.style.use('seaborn-darkgrid')

# Create the line plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each faculty type as a separate series
for faculty in melted_data['faculty_type'].unique():
    subset = melted_data[melted_data['faculty_type'] == faculty]
    ax.plot(subset['Year'], subset['Percentage'], marker='o', label=faculty)

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Employment Percentage (%)')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 70 with threat_id: thread_K8R3P4Re7gW2PepwRg5XI1o5
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe from wide to long format
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Set the plot style
plt.style.use('seaborn-darkgrid')

# Create the line plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each faculty type as a separate series
for faculty in melted_data['faculty_type'].unique():
    subset = melted_data[melted_data['faculty_type'] == faculty]
    ax.plot(subset['Year'], subset['Percentage'], marker='o', label=faculty)

# Add labels and improve them
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Employment Percentage (%)', fontsize=12)
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14, weight='bold')
ax.legend(title='Faculty Type', fontsize=10, title_fontsize='11')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Add a grid for ease of reading
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Move the legend outside of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Ensure layout is tight
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 71 with threat_id: thread_zkTTjgvM1HAQAI5GjPBmdmyE
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(10, 6))

# Define the years to plot
years = list(data.columns[1:])

# Plot each faculty type
for index, row in data.iterrows():
    plt.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Add labels, legend, and show plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Faculty')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 71 with threat_id: thread_zkTTjgvM1HAQAI5GjPBmdmyE
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(12, 8))

# Define the years to plot
years = list(data.columns[1:])

# Plot each faculty type with a filled area
for index, row in data.iterrows():
    plt.plot(years, row[1:], marker='o', label=row['faculty_type'])
    plt.fill_between(years, row[1:], alpha=0.2)

# Add labels, legend, and show plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Faculty')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 71 with threat_id: thread_zkTTjgvM1HAQAI5GjPBmdmyE
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(12, 8))

# Define the years to plot
years = list(data.columns[1:])

# Plot each faculty type with a filled area
for index, row in data.iterrows():
    plt.plot(years, row[1:], marker='o', label=row['faculty_type'], linestyle='-', linewidth=2)
    plt.fill_between(years, row[1:], alpha=0.15)

# Add labels, legend, and show plot with improved styling
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Total Faculty (%)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Faculty Type', loc='upper right', fontsize=12, title_fontsize='13')
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 72 with threat_id: thread_fsZwxq3OUe5tr7CbU8kF6j1Z
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for easier plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 72 with threat_id: thread_fsZwxq3OUe5tr7CbU8kF6j1Z
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for easier plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_melted, x='year', y='percentage', hue='faculty_type', marker='o')
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Faculty Type')
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 72 with threat_id: thread_fsZwxq3OUe5tr7CbU8kF6j1Z
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for easier plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_melted, x='year', y='percentage', hue='faculty_type', marker='o', linewidth=2.5)
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(right=0.8)  # Adjust plot to provide space for the legend
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 73 with threat_id: thread_Cc869w6dMvcWz65aT9Dsjtke
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Set the style of seaborn
sns.set(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data_melted, x='year', y='percentage', hue='faculty_type', style='faculty_type', s=100)

# Enhance the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 73 with threat_id: thread_Cc869w6dMvcWz65aT9Dsjtke
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert the 'year' column to numeric
data_melted['year'] = data_melted['year'].astype(int)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create the line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=data_melted, x='year', y='percentage', hue='faculty_type', marker='o')

# Enhance the plot
plt.title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 73 with threat_id: thread_Cc869w6dMvcWz65aT9Dsjtke
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data to long format for plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Convert the 'year' column to numeric
data_melted['year'] = data_melted['year'].astype(int)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create the line plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_melted, x='year', y='percentage', hue='faculty_type', marker='o')

# Enhance the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=18, pad=20)
plt.xlabel('Year', fontsize=14, labelpad=15)
plt.ylabel('Employment Percentage (%)', fontsize=14, labelpad=15)

# Customize ticks and rotation
plt.xticks(rotation=45)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

# Move the legend to a clearer position
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='12', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Show a grid for better readability
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Adjust layout to accommodate everything neatly
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 74 with threat_id: thread_uH5eBuEMA5HWRFpIlCuIZczL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the uploaded file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Transform the data into long-form format
long_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert the 'Year' column to numeric
long_data['Year'] = pd.to_numeric(long_data['Year'])

# Set plot size
plt.figure(figsize=(12, 8))

# Create the dot plot
sns.pointplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', palette='Set1', markers='o')

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.ylabel('Percentage [%]')
plt.xlabel('Year')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 74 with threat_id: thread_uH5eBuEMA5HWRFpIlCuIZczL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming long_data is already prepared from the previous step
# Set plot style and size
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Create a line plot with markers
sns.lineplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', palette='Set2', marker='o')

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.ylabel('Percentage [%]')
plt.xlabel('Year')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 74 with threat_id: thread_uH5eBuEMA5HWRFpIlCuIZczL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming long_data is already prepared from the previous step
# Set plot style and size
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))

# Create a line plot with markers
palette = sns.color_palette("husl", len(data['faculty_type'].unique()))
sns.lineplot(data=long_data, x='Year', y='Percentage', hue='faculty_type', palette=palette, marker='o', linewidth=2)

# Customize the plot
plt.title('Trends in Instructional Staff Employment by Faculty Type (1975-2011)', fontsize=16, weight='bold')
plt.ylabel('Percentage of Total Instructional Staff (%)', fontsize=12, weight='bold')
plt.xlabel('Year', fontsize=12, weight='bold')
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)

# Customize legend
plt.legend(
    title='Faculty Type', 
    title_fontsize='13', 
    fontsize='11', 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left'
)

# Add grid lines for clarity
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Ensure the layout is tight
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 76 with threat_id: thread_Z2HezLc0poFNQejBybWVrOS5
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the years and types for plotting
years = data.columns[1:]
types = data['faculty_type']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate each faculty type for plotting
for index, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Customize the plot
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Employment Percentage')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 76 with threat_id: thread_Z2HezLc0poFNQejBybWVrOS5
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the years for plotting
years = data.columns[1:]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Create an area plot
data.set_index('faculty_type').T.plot.area(ax=ax, legend=True)

# Customize the plot
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Employment Percentage')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 76 with threat_id: thread_Z2HezLc0poFNQejBybWVrOS5
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the years for plotting
years = data.columns[1:]

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))

# Create an area plot with improved aesthetics
data.set_index('faculty_type').T.plot.area(ax=ax, alpha=0.7, cmap='viridis')

# Customize the plot
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Year', fontsize=12, labelpad=10)
ax.set_ylabel('Employment Percentage (%)', fontsize=12, labelpad=10)
ax.legend(title='Faculty Type', title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Set tick parameters
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Improve layout
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 77 with threat_id: thread_yDRIf38MPcnnzSduC5LuuMFK
import matplotlib.pyplot as plt
import seaborn as sns

# Set the seaborn theme for the plot
sns.set_theme(style="whitegrid")

# Create a line plot
plt.figure(figsize=(14, 8))
ax = sns.lineplot(data=melted_data, x="Year", y="Percentage", hue="faculty_type", marker='o', linewidth=2.5)

# Title and labels
plt.title("Trends in Instructional Staff Employment Over Time", fontsize=18, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Percentage of Total Faculty", fontsize=14)

# Axis ticks
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Legend
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)

# Grid
plt.grid(visible=True, linestyle='--', linewidth=0.5)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 78 with threat_id: thread_J6Hne04nTYSIYsHS2Sgx2Wze
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(12, 8))

# Iterate through each faculty type to plot their data
for index, row in data.iterrows():
    plt.plot(row.index[1:], row.values[1:], 'o-', label=row['faculty_type'])

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage/Number')
plt.title('Instructional Staff Employment Trends')
plt.legend(title='Faculty Type', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 78 with threat_id: thread_J6Hne04nTYSIYsHS2Sgx2Wze
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(14, 8))

# Iterate through each faculty type to plot their data
for index, row in data.iterrows():
    # Plot a line for each faculty type
    plt.plot(row.index[1:], row.values[1:], marker='o', label=row['faculty_type'])
    # Emphasize each point as a bar
    plt.bar(row.index[1:], row.values[1:], alpha=0.3, label=f"{row['faculty_type']} Bar")

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage/Number')
plt.title('Instructional Staff Employment Trends')
plt.legend(title='Faculty Type', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 78 with threat_id: thread_J6Hne04nTYSIYsHS2Sgx2Wze
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set up the plot
plt.figure(figsize=(14, 8))

# Define colors for differentiation
colors = plt.cm.tab10.colors

# Plot each faculty type
for index, row in data.iterrows():
    plt.plot(row.index[1:], row.values[1:], marker='o', color=colors[index], label=row['faculty_type'])
    plt.bar(row.index[1:], row.values[1:], alpha=0.1, color=colors[index])

# Add detailed labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage/Number of Staff', fontsize=12)
plt.title('Instructional Staff Employment Trends over Years', fontsize=16)

# Set y-axis limit if necessary
plt.ylim(0, max(data.values[:, 1:].max(axis=1)) * 1.15)

# Enhance the legend
plt.legend(title='Faculty Type', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

# Activate grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Improve x-axis ticks and rotation for better visibility
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 79 with threat_id: thread_ri4l3SQJCVdtIw0CZ5xTig3p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Creating a sample dataset
data = {
    'Year': [2018, 2019, 2020, 2021],
    'Staff': [250, 270, 300, 320],
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# Create a dot plot
plt.plot(df['Year'], df['Staff'], 'o-', label='Instructional Staff')

# Adding improvements and labels
plt.title('Instructional Staff Employment Trends', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Staff', fontsize=12)
plt.xticks(df['Year'])  # Set x-ticks based on the year
plt.yticks(np.arange(240, 340, 20))  # Set y-ticks for better visualization
plt.grid(True)
plt.legend(title='Legend')
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 80 with threat_id: thread_y6SjnCQHvf9rV6X0uv8aBV5N
import matplotlib.pyplot as plt

# Ensure plot can display inline
%matplotlib inline

# Data preparation
faculty_types = data['faculty_type']
years = data.columns[1:]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over each faculty type and plot its trend over the years
for index, row in data.iterrows():
    ax.plot(years, row[1:], marker='o', label=row['faculty_type'])

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.set_title('Instructional Staff Employment Trends')
ax.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.1, Round 80 with threat_id: thread_y6SjnCQHvf9rV6X0uv8aBV5N
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
years = data.columns[1:]
indices = np.arange(len(years))

# Set plot size
fig, ax = plt.subplots(figsize=(12, 8))

# Bar width
bar_width = 0.15

# Colors for different faculty types
colors = ['b', 'g', 'r', 'c', 'm']

# Iterating over each faculty type to create a bar for each
for i, (index, row) in enumerate(data.iterrows()):
    ax.bar(indices + i * bar_width, row[1:], bar_width, label=row['faculty_type'], color=colors[i])

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.set_title('Instructional Staff Employment Trends')
ax.set_xticks(indices + bar_width * 2)
ax.set_xticklabels(years)
ax.legend(title='Faculty Type')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.2, Round 80 with threat_id: thread_y6SjnCQHvf9rV6X0uv8aBV5N
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
years = data.columns[1:]
indices = np.arange(len(years))

# Set plot size
fig, ax = plt.subplots(figsize=(12, 8))

# Bar width
bar_width = 0.15

# Colors for different faculty types
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create a bar for each faculty type
for i, (index, row) in enumerate(data.iterrows()):
    ax.bar(indices + i * bar_width, row[1:], bar_width, label=row['faculty_type'], color=colors[i])

# Adding labels and title with improvements
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of Employment (%)', fontsize=12, fontweight='bold')
ax.set_title('Trends in Instructional Staff Employment (1975-2011)', fontsize=14, fontweight='bold')
ax.set_xticks(indices + bar_width * 2)
ax.set_xticklabels(years, fontsize=10)
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)

# Improve legend position and appearance
ax.legend(title='Faculty Type', title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))

# Adjust layout to fit components neatly
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 81 with threat_id: thread_6LAX5Mra8Zv4TPynMKxG2LCF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the style for the plot
sns.set(style="whitegrid")

# Melt the data to make it suitable for dot plotting
melted_data = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Plotting the data using seaborn
plt.figure(figsize=(14, 8))
sns.scatterplot(data=melted_data, x="Year", y="Percentage", hue="faculty_type", style="faculty_type", s=100)

# Adding titles and labels
plt.title("Instructional Staff Employment Trends (1975-2011)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Employment Percentage (%)", fontsize=14)
plt.legend(title="Faculty Type", fontsize=12, title_fontsize='13')

# Display the plot
plt.show()
##################################################
#Question 14.1, Round 81 with threat_id: thread_6LAX5Mra8Zv4TPynMKxG2LCF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the style for the plot
sns.set(style="whitegrid")

# Melt the data to make it suitable for line plotting
melted_data = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Convert Year to integer for proper plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Plotting the data using seaborn
plt.figure(figsize=(14, 8))
sns.lineplot(data=melted_data, x="Year", y="Percentage", hue="faculty_type", marker="o", linewidth=2.5)

# Adding titles and labels
plt.title("Instructional Staff Employment Trends (1975-2011)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Employment Percentage (%)", fontsize=14)
plt.legend(title="Faculty Type", fontsize=12, title_fontsize='13')

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 81 with threat_id: thread_6LAX5Mra8Zv4TPynMKxG2LCF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Set the style and color palette for the plot
sns.set(style="whitegrid")
palette = sns.color_palette("husl", len(data['faculty_type'].unique()))

# Melt the data for plotting
melted_data = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

# Convert Year to integer for proper plotting
melted_data['Year'] = melted_data['Year'].astype(int)

# Plotting the data
plt.figure(figsize=(16, 9))
sns.lineplot(
    data=melted_data,
    x="Year",
    y="Percentage",
    hue="faculty_type",
    style="faculty_type",
    markers=True,
    dashes=False,
    palette=palette,
    linewidth=2.5
)

# Add titles and labels with improved formatting
plt.title("Trends of Instructional Staff Employment (1975-2011)", 
          fontsize=18, weight='bold')
plt.xlabel("Year", fontsize=15)
plt.ylabel("Employment Percentage (%)", fontsize=15)

# Customize legend
plt.legend(title="Type of Faculty", fontsize=12, title_fontsize='13', loc='upper right')

# Enhance the plot's aesthetics
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linewidth=0.6, linestyle='--', alpha=0.7) 

# Display the plot
plt.tight_layout(pad=2)
plt.show()
##################################################
#Question 14.0, Round 83 with threat_id: thread_adFTg6HNi2v9YklfbVShFKSp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years and employment as columns
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Employment')

# Plotting
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create a dot plot
sns.scatterplot(x='Year', y='Employment', hue='faculty_type', style='faculty_type',
                s=100, palette='deep', data=data_melted)

# Improving the plot
plt.title('Instructional Staff Employment Trends')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 83 with threat_id: thread_adFTg6HNi2v9YklfbVShFKSp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years and employment as columns
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Employment')

# Plotting
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create a line plot
sns.lineplot(x='Year', y='Employment', hue='faculty_type', marker='o', data=data_melted, palette='deep')

# Improving the plot
plt.title('Instructional Staff Employment Trends')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 83 with threat_id: thread_adFTg6HNi2v9YklfbVShFKSp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataframe to have years and employment as columns
data_melted = data.melt(id_vars='faculty_type', var_name='Year', value_name='Employment')

# Plotting
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid", context="talk")

# Create a line plot
sns.lineplot(x='Year', y='Employment', hue='faculty_type', marker='o', data=data_melted, palette='tab10')

# Improving the plot
plt.title('Trends in Instructional Staff Employment by Faculty Type (1975-2011)', fontsize=16, weight='bold')
plt.xticks(rotation=45)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Total Employment (%)', fontsize=14)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Show grid lines more prominently
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.7)

plt.show()
##################################################
#Question 14.0, Round 87 with threat_id: thread_zFb1O5lRLvzFpx1WjPK11UVT
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (assuming it's stored as 'data' DataFrame)
# data = pd.read_csv('path_to_your_file.csv')

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot each faculty type using a different marker for distinction
for index, row in data.iterrows():
    plt.plot(data.columns[1:], row[1:], marker='o', markersize=8, label=row['faculty_type'])

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(range(0, 50, 5))  # Set range of y-axis
plt.grid(True, linestyle='--', alpha=0.5)

# Add a legend to identify faculty types
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()
##################################################
#Question 14.1, Round 87 with threat_id: thread_zFb1O5lRLvzFpx1WjPK11UVT
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (assuming it's stored as 'data' DataFrame)
# data = pd.read_csv('path_to_your_file.csv')

# Set the figure size
plt.figure(figsize=(12, 8))

# Prepare data for area plot by extracting columns for each year
years = data.columns[1:].astype(int)
faculty_types = data['faculty_type']

# Transpose the data to get years as rows for plotting
area_data = data.set_index('faculty_type').T

# Plot stacked area chart
plt.stackplot(years, [area_data[faculty] for faculty in faculty_types], labels=faculty_types, alpha=0.8)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Instructional Staff Employment Trends (Stacked Area Plot)')
plt.xticks(years, rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(range(0, 110, 10))  # Set range of y-axis to maximum of 100
plt.grid(True, linestyle='--', alpha=0.5)

# Add a legend to identify faculty types
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()
##################################################
#Question 14.2, Round 87 with threat_id: thread_zFb1O5lRLvzFpx1WjPK11UVT
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (assuming it's stored as 'data' DataFrame)
# data = pd.read_csv('path_to_your_file.csv')

# Set the figure size
plt.figure(figsize=(14, 8))

# Define colors for each faculty type for better differentiation
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

# Prepare data for area plot by extracting columns for each year
years = data.columns[1:].astype(int)
faculty_types = data['faculty_type']

# Transpose the data to get years as rows for plotting
area_data = data.set_index('faculty_type').T

# Plot stacked area chart
plt.stackplot(years, [area_data[faculty] for faculty in faculty_types], labels=faculty_types, colors=colors, alpha=0.8)

# Add labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage (%) of Total Employment', fontsize=12)
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold')
plt.xticks(years, rotation=45, fontsize=10)  # Rotate x-axis labels for better readability
plt.yticks(range(0, 110, 10), fontsize=10)  # Set range of y-axis to maximum of 100

# Add a legend to identify faculty types properly
plt.legend(title='Faculty Type', loc='upper left', fontsize=10, title_fontsize='12', bbox_to_anchor=(1, 1))

# Add grid for better visual separation
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Add annotations for clarity if needed, here is an example (uncomment if desired)
# plt.annotate('Some notable trend', xy=(2005, 30), xytext=(1990, 35),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# Show the plot
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()
##################################################
#Question 14.0, Round 88 with threat_id: thread_axLODEBcxGeXwUoOkQEpEWlO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data to a long format
data_long = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Employment_percentage')

# Convert the 'Year' column to numeric
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Year', y='Employment_percentage', hue='faculty_type', data=data_long, s=100)

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 88 with threat_id: thread_axLODEBcxGeXwUoOkQEpEWlO
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data to a long format
data_long = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Employment_percentage')

# Convert the 'Year' column to numeric
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Plot using matplotlib
plt.figure(figsize=(12, 8))
for faculty_type in data_long['faculty_type'].unique():
    plt.plot(
        data_long[data_long['faculty_type'] == faculty_type]['Year'],
        data_long[data_long['faculty_type'] == faculty_type]['Employment_percentage'],
        marker='o',
        label=faculty_type
    )

# Customize the plot
plt.title('Instructional Staff Employment Trends (1975-2011)')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 88 with threat_id: thread_axLODEBcxGeXwUoOkQEpEWlO
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Reshape the data to a long format
data_long = pd.melt(data, id_vars=['faculty_type'], var_name='Year', value_name='Employment_percentage')

# Convert the 'Year' column to numeric
data_long['Year'] = pd.to_numeric(data_long['Year'])

# Select a plotting style
plt.style.use('seaborn-colorblind')

# Plot using matplotlib
plt.figure(figsize=(14, 9))
for faculty_type in data_long['faculty_type'].unique():
    plt.plot(
        data_long[data_long['faculty_type'] == faculty_type]['Year'],
        data_long[data_long['faculty_type'] == faculty_type]['Employment_percentage'],
        marker='o',
        label=faculty_type
    )

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.xticks(range(1975, 2012, 4))  # Set x-ticks for better readability
plt.yticks(fontsize=12)
plt.legend(title='Faculty Type', title_fontsize=12, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 90 with threat_id: thread_zgWkjGWVnpuQcG1wUVmKosM5
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV file
data = pd.read_csv('/path/to/your/file.csv')

# Transform the dataframe to a format compatible with plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(12, 8))
for faculty in data['faculty_type']:
    subset = data_melted[data_melted['faculty_type'] == faculty]
    plt.plot(subset['year'], subset['percentage'], marker='o', linestyle='-', label=faculty)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type')
plt.grid(visible=True)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 90 with threat_id: thread_zgWkjGWVnpuQcG1wUVmKosM5
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV file
data = pd.read_csv('/path/to/your/file.csv')

# Transform the dataframe to a format compatible with plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plotting
plt.figure(figsize=(14, 8))
faculty_types = data['faculty_type'].unique()
years = data.columns[1:]

# Set bar width
bar_width = 0.15

# Create an index for each year
r = range(len(years))

# Plot each faculty type
for i, faculty in enumerate(faculty_types):
    values = data.loc[data['faculty_type'] == faculty, years].values.flatten()
    plt.bar([x + i * bar_width for x in r], values, width=bar_width, label=faculty)

plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks([x + bar_width for x in r], years, rotation=45)
plt.legend(title='Faculty Type')
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 90 with threat_id: thread_zgWkjGWVnpuQcG1wUVmKosM5
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from the CSV file
data = pd.read_csv('/path/to/your/file.csv')

# Transform the dataframe to a format compatible with plotting
data_melted = data.melt(id_vars='faculty_type', var_name='year', value_name='percentage')

# Plot configuration
plt.figure(figsize=(14, 8))
faculty_types = data['faculty_type'].unique()
years = data.columns[1:]

# Set bar width
bar_width = 0.15

# Create an index for each year
r = np.arange(len(years))

# Add zero line for context
plt.axhline(0, color='gray', linewidth=0.8)

# Plot each faculty type
for i, faculty in enumerate(faculty_types):
    values = data.loc[data['faculty_type'] == faculty, years].values.flatten()
    plt.bar(r + i * bar_width, values, width=bar_width, label=faculty)

# Title and labels
plt.title('Trends in Instructional Staff Employment', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage', fontsize=12)

# X-ticks and adjustments
plt.xticks(r + bar_width * (len(faculty_types) - 1) / 2, years, rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Legend
plt.legend(title='Faculty Type', fontsize=10)

# Grid and layout
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 91 with threat_id: thread_0eDwb2RcdP5N4ZCNLsOoBv4S
import matplotlib.pyplot as plt

# Set the style of the plot
plt.style.use('seaborn-whitegrid')

# Colors for different faculty types
colors = ['r', 'b', 'g', 'c', 'm']

# Plot each faculty type
fig, ax = plt.subplots(figsize=(12, 8))
years = data.columns[1:].astype(int)  # Convert year columns to integers

for i, row in data.iterrows():
    ax.plot(years, row[1:], '-o', label=row['faculty_type'], color=colors[i])

# Set plot title and labels
ax.set_title('Instructional Staff Employment Trends', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.legend(title='Faculty Type')

# Show the plot
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 91 with threat_id: thread_0eDwb2RcdP5N4ZCNLsOoBv4S
import matplotlib.pyplot as plt

# Prepare the data
years = data.columns[1:].astype(int)  # Convert year columns to integers
faculty_types = data['faculty_type']
percentages = data.iloc[:, 1:].values  # Extract percentage values

# Colors for the stacked area plot
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']

# Create the stacked area plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.stackplot(years, percentages, labels=faculty_types, colors=colors, alpha=0.8)

# Set plot title and labels
ax.set_title('Instructional Staff Employment Trends', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.legend(title='Faculty Type', loc='upper left')

# Show the plot
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 91 with threat_id: thread_0eDwb2RcdP5N4ZCNLsOoBv4S
import matplotlib.pyplot as plt

# Prepare the data
years = data.columns[1:].astype(int)  # Convert year columns to integers
faculty_types = data['faculty_type']
percentages = data.iloc[:, 1:].values  # Extract percentage values

# Colors for the stacked area plot
colors = ['#c00000', '#00b0f0', '#92d050', '#ffc000', '#7030a0']

# Create the stacked area plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.stackplot(years, percentages, labels=faculty_types, colors=colors, alpha=0.8)

# Improve plot title and labels
ax.set_title('Instructional Staff Employment Trends (1975-2011)', fontsize=18, fontweight='bold')
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Percentage of Total Faculty (%)', fontsize=14)

# Improve legend appearance
ax.legend(title='Faculty Type', loc='upper left', fontsize=12, title_fontsize=14)

# Improve grid and ticks
ax.grid(True, linestyle='--', alpha=0.5)
plt.xticks(years, rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.0, Round 93 with threat_id: thread_PpKo8dt0vwF1XhlTt74HP0Wl
import matplotlib.pyplot as plt

# Extract the years and faculty types
years = data.columns[1:]
faculty_types = data['faculty_type']

# Create a dot plot
plt.figure(figsize=(12, 6))

for index, faculty_type in enumerate(faculty_types):
    # Plot dots for each year for a particular faculty type
    plt.plot(years, data.iloc[index, 1:], 'o-', label=faculty_type)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('Instructional Staff Employment Trends')
plt.legend(title='Faculty Type')
plt.grid(True)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 93 with threat_id: thread_PpKo8dt0vwF1XhlTt74HP0Wl
import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Extract the years as integer type for plotting
years = data.columns[1:].astype(int)

# Transpose the data for an area plot
area_data = data.drop(columns='faculty_type').T
area_data.columns = data['faculty_type']

# Create an area plot
plt.figure(figsize=(12, 6))
area_data.plot(kind='area', stacked=True, alpha=0.5)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('Instructional Staff Employment Trends (Area Plot)')
plt.grid(True)
plt.xticks(ticks=range(len(years)), labels=years, rotation=45)

# Show legend and plot
plt.legend(title='Faculty Type', loc='upper left')
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 93 with threat_id: thread_PpKo8dt0vwF1XhlTt74HP0Wl
import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded file
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Extract the years as integer type for plotting
years = data.columns[1:].astype(int)

# Transpose the data for an area plot
area_data = data.drop(columns='faculty_type').T
area_data.columns = data['faculty_type']

# Create an area plot
fig, ax = plt.subplots(figsize=(12, 6))
area_data.plot(kind='area', stacked=True, alpha=0.7, ax=ax, colormap='Accent')

# Add labels and title with improvements
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Percentage of Employment (%)', fontsize=12)
ax.set_title('Trends in Instructional Staff Employment Over Time', fontsize=14, weight='bold')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Improve x-ticks for clarity
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, rotation=45, ha='right')

# Improve the legend
ax.legend(title='Faculty Type', title_fontsize='13', fontsize='11', loc='upper left')

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 94 with threat_id: thread_VYDlsOakqfd2qB9oznbkfm63
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Plotting the instructional staff employment trends as a dot plot
def plot_dot(data, x_col, y_col, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_col], data[y_col], color='royalblue')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Assuming 'Year' and 'Employment' are the columns of interest
plot_dot(df, 'Year', 'Employment', 'Instructional Staff Employment Trends')
##################################################
#Question 14.1, Round 94 with threat_id: thread_VYDlsOakqfd2qB9oznbkfm63
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Plotting the instructional staff employment trends as a line plot
def plot_line(data, x_col, y_col, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], marker='o', linestyle='-', color='darkorange')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Assuming 'Year' and 'Employment' are the columns of interest
plot_line(df, 'Year', 'Employment', 'Instructional Staff Employment Trends')
##################################################
#Question 14.2, Round 94 with threat_id: thread_VYDlsOakqfd2qB9oznbkfm63
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe for reference
print(df.head())

# Improved plotting function
def plot_line_improved(data, x_col, y_col, title, x_label, y_label):
    plt.figure(figsize=(12, 7))
    plt.plot(data[x_col], data[y_col], marker='o', linestyle='-', color='darkorange', label='Employment Trend')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()

# Assuming 'Year' and 'Employment' are the correct column names. Adjust if needed.
plot_line_improved(df, 'Year', 'Employment', 
                   'Instructional Staff Employment Trends Over Time', 
                   'Year', 
                   'Number of Employed Staff')
##################################################
#Question 14.0, Round 95 with threat_id: thread_FsPuJFjhEEpYQQHL5xkqwlcM
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set(style="whitegrid")

# Convert the dataset from wide to long format for easier plotting with seaborn
data_long = data.melt(id_vars=["faculty_type"], var_name="Year", value_name="Employment")

# Initialize the matplotlib figure
plt.figure(figsize=(14, 8))

# Create a dot plot
sns.scatterplot(data=data_long, x="Year", y="Employment", hue="faculty_type", style="faculty_type", s=100)

# Add title and labels
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Percentage of Employment")
plt.xticks(rotation=45)

# Show legend
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 95 with threat_id: thread_FsPuJFjhEEpYQQHL5xkqwlcM
import matplotlib.pyplot as plt

# Set the plot style
plt.style.use('seaborn-darkgrid')

# Initialize the matplotlib figure
plt.figure(figsize=(14, 8))

# Plot each faculty type as an area in the plot
for idx, faculty_type in enumerate(data['faculty_type']):
    plt.fill_between(data.columns[1:], data.iloc[idx, 1:], label=faculty_type, alpha=0.6)

# Add title and labels
plt.title("Instructional Staff Employment Trends")
plt.xlabel("Year")
plt.ylabel("Percentage of Employment")
plt.xticks(rotation=45)

# Show legend
plt.legend(title="Faculty Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 95 with threat_id: thread_FsPuJFjhEEpYQQHL5xkqwlcM
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style and color palette
plt.style.use('seaborn-darkgrid')
colors = sns.color_palette("muted")

# Initialize the matplotlib figure
plt.figure(figsize=(14, 8))

# Plot each faculty type as an area
for idx, (faculty_type, color) in enumerate(zip(data['faculty_type'], colors)):
    plt.fill_between(data.columns[1:], data.iloc[idx, 1:], label=faculty_type, alpha=0.7, color=color)

# Add title and axis labels with improved descriptions
plt.title("Trends in Instructional Staff Employment by Faculty Type (1975-2011)", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Percentage of Employment (%)", fontsize=14)

# Enhance x-axis with better rotation for readability and improved tick marks
plt.xticks(rotation=45)
plt.xlim(data.columns[1:].min(), data.columns[1:].max())

# Adjust y-axis limit
plt.ylim(0, 50)

# Improve legend placement and appearance
plt.legend(title="Faculty Type", title_fontsize='13', fontsize='11', bbox_to_anchor=(1.01, 1), loc='upper left')

# Tweak layout for better appearance
plt.tight_layout()

# Show plot
plt.show()
##################################################
#Question 14.0, Round 96 with threat_id: thread_ZH0PI1RHaseYseqwjEn3Xgmv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the function to plot the trends
def plot_instructional_staff_trends(data):
    # Melting the data to long format
    data_long = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

    # Convert the year to integer for sorting purposes
    data_long["Year"] = data_long["Year"].astype(int)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Create a dot plot
    dot_plot = sns.scatterplot(data=data_long, x="Year", y="faculty_type", size="Percentage", sizes=(20, 200), legend=False, hue="faculty_type", palette="tab10")

    # Adding informative labels
    plt.title("Instructional Staff Employment Trends (1975-2011)", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Faculty Type", fontsize=12)
    plt.xticks(range(1975, 2012, 4))  # Showing every 4 years for better readability
    plt.grid(True)
    plt.show()

# Run the plot function
plot_instructional_staff_trends(data)
##################################################
#Question 14.1, Round 96 with threat_id: thread_ZH0PI1RHaseYseqwjEn3Xgmv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Define the function for line plot
def plot_instructional_staff_trends_line(data):
    # Melting the data to long format
    data_long = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

    # Convert the year to integer for sorting purposes
    data_long["Year"] = data_long["Year"].astype(int)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Create a line plot
    line_plot = sns.lineplot(data=data_long, x="Year", y="Percentage", hue="faculty_type", palette="tab10", marker='o')

    # Adding informative labels
    plt.title("Instructional Staff Employment Trends (1975-2011) - Line Plot", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.xticks(range(1975, 2012, 4))  # Showing every 4 years for better readability
    plt.grid(True)
    plt.legend(title="Faculty Type", fontsize=10, title_fontsize='13', loc='upper right')
    plt.show()

# Run the line plot function
plot_instructional_staff_trends_line(data)
##################################################
#Question 14.2, Round 96 with threat_id: thread_ZH0PI1RHaseYseqwjEn3Xgmv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

def plot_instructional_staff_trends_line_improved(data):
    # Melting the data to long format
    data_long = data.melt(id_vars="faculty_type", var_name="Year", value_name="Percentage")

    # Convert the year to integer for sorting purposes
    data_long["Year"] = data_long["Year"].astype(int)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Create a line plot
    line_plot = sns.lineplot(data=data_long, x="Year", y="Percentage", hue="faculty_type", palette="tab10", marker='o')

    # Adding informative labels with improved readability
    plt.title("Trends in Instructional Staff Employment (1975-2011)", fontsize=18, fontweight='bold')
    plt.xlabel("Year", fontsize=14, fontweight='bold')
    plt.ylabel("Percentage of Employment (%)", fontsize=14, fontweight='bold')
    plt.xticks(range(1975, 2012, 4), fontsize=12)  # Showing every 4 years for better readability
    plt.yticks(fontsize=12)
    
    # Add gridlines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Legend improvements
    plt.legend(title="Faculty Type", fontsize=12, title_fontsize='14', loc='upper right')

    # Title for x and y axis grids
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# Run the improved line plot function
plot_instructional_staff_trends_line_improved(data)
##################################################
#Question 14.0, Round 97 with threat_id: thread_ALtK80OSlSx087xauNk9Sk2e
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create the dot plot
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Plot each faculty type in the dataset
sns.scatterplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', s=100, palette='deep')

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 97 with threat_id: thread_ALtK80OSlSx087xauNk9Sk2e
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Create a line plot
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Plot each faculty type in the dataset
sns.lineplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', marker='o', palette='deep')

# Customize the plot
plt.title('Instructional Staff Employment Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.2, Round 97 with threat_id: thread_ALtK80OSlSx087xauNk9Sk2e
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the data for plotting
melted_data = data.melt(id_vars='faculty_type', var_name='Year', value_name='Percentage')

# Convert 'Year' to numeric for proper plotting
melted_data['Year'] = pd.to_numeric(melted_data['Year'])

# Create a line plot
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Plot each faculty type in the dataset
sns.lineplot(data=melted_data, x='Year', y='Percentage', hue='faculty_type', marker='o', palette='deep')

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975 - 2011)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Employment Percentage (%)', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(melted_data['Year'].min(), melted_data['Year'].max())

# Enhance legend
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# Improve layout
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()
##################################################
#Question 14.0, Round 98 with threat_id: thread_yDoCilIPEBnZJ8VAJw3zrKoB
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(file_path)

# Melt the dataset to have years as a variable
data_melted = pd.melt(data, id_vars=['faculty_type'], 
                      var_name='year', value_name='percentage')

# Create a dot plot
plt.figure(figsize=(10, 6))
for faculty_type, group_data in data_melted.groupby('faculty_type'):
    plt.plot(group_data['year'], group_data['percentage'], 
             marker='o', linestyle='', label=faculty_type)

# Customize the plot
plt.title('Instructional Staff Employment Trends')
plt.xlabel('Year')
plt.ylabel('Employment Percentage')
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)

# Show the plot
plt.show()
##################################################
#Question 14.1, Round 98 with threat_id: thread_yDoCilIPEBnZJ8VAJw3zrKoB
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataset to have years as a variable
data_melted = pd.melt(data, id_vars=['faculty_type'], 
                      var_name='year', value_name='percentage')

# Create a line plot with markers
plt.figure(figsize=(12, 8))
for faculty_type, group_data in data_melted.groupby('faculty_type'):
    plt.plot(group_data['year'], group_data['percentage'], 
             marker='o', linestyle='-', label=faculty_type)

# Customize the plot
plt.title('Instructional Staff Employment Trends', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Employment Percentage', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Faculty Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.2, Round 98 with threat_id: thread_yDoCilIPEBnZJ8VAJw3zrKoB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/file-5riwCDAHXme7U6mfNZwCLe'
data = pd.read_csv(file_path)

# Melt the dataset to have years as a variable
data_melted = pd.melt(data, id_vars=['faculty_type'], 
                      var_name='year', value_name='percentage')

# Set a style for the plot
sns.set(style="whitegrid")

# Create a line plot with markers
plt.figure(figsize=(14, 8))
colors = sns.color_palette("husl", len(data['faculty_type'].unique()))

for (faculty_type, group_data), color in zip(data_melted.groupby('faculty_type'), colors):
    plt.plot(group_data['year'], group_data['percentage'], 
             marker='o', linestyle='-', label=faculty_type, color=color)

# Customize the plot
plt.title('Trends in Instructional Staff Employment (1975-2011)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Total Staff (%)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Faculty Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
##################################################
#Question 14.0, Round 99 with threat_id: thread_NfoTS2J2HPVklxvFrgDwrWAC
import pandas as pd
import matplotlib.pyplot as plt

# Data loading (this line should be customized based on your file location)
data = pd.read_csv('/path/to/your/file.csv')  # Replace with the actual path

# Preparing the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Years to be plotted
years = data.columns[1:]

# Plot each faculty type
for i, row in data.iterrows():
    ax.plot(years, row[1:], 'o-', label=row['faculty_type'])

# Adding labels and title
ax.set_title('Instructional Staff Employment Trends')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
##################################################
#Question 14.1, Round 99 with threat_id: thread_NfoTS2J2HPVklxvFrgDwrWAC
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data loading (this line should be customized based on your file location)
data = pd.read_excel('/path/to/your/file.xlsx')  # Replace with the actual path

# Setup the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Extract the years and faculty types
years = data.columns[1:]
faculty_types = data['faculty_type']

# Determine the number of groups and bar widths
num_years = len(years)
bar_width = 0.15

# X locations for the groups
indices = np.arange(len(faculty_types))

# Plot each group of bars
for i, year in enumerate(years):
    ax.bar(indices + i * bar_width, data[year], bar_width, label=year)

# Adding labels and title
ax.set_title('Instructional Staff Employment Trends (Grouped Bar Plot)')
ax.set_xlabel('Faculty Type')
ax.set_ylabel('Percentage')
ax.set_xticks(indices + bar_width * (num_years - 1) / 2)
ax.set_xticklabels(faculty_types, rotation=45)
ax.legend(title='Year')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()
##################################################
#Question 14.2, Round 99 with threat_id: thread_NfoTS2J2HPVklxvFrgDwrWAC
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data loading (this line should be customized based on your file location)
data = pd.read_excel('/path/to/your/file.xlsx')  # Replace with the actual path

# Setup the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Extract the years and faculty types
years = data.columns[1:]
faculty_types = data['faculty_type']

# Determine the number of groups and bar widths
num_years = len(years)
bar_width = 0.15

# X locations for the groups
indices = np.arange(len(faculty_types))

# Create color map
colors = plt.cm.viridis(np.linspace(0, 1, num_years))

# Plot each group of bars
for i, (year, color) in enumerate(zip(years, colors)):
    ax.bar(indices + i * bar_width, data[year], bar_width, label=year, color=color)

# Adding labels and title
ax.set_title('Instructional Staff Employment Trends (1975-2011)', fontsize=16)
ax.set_xlabel('Faculty Type', fontsize=14)
ax.set_ylabel('Employment Percentage (%)', fontsize=14)
ax.set_xticks(indices + bar_width * (num_years - 1) / 2)
ax.set_xticklabels(faculty_types, rotation=45, ha='right', fontsize=12)
ax.legend(title='Year', title_fontsize='13', fontsize='11')

# Grid and layout adjustments
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(pad=2)

# Display the plot
plt.show()
