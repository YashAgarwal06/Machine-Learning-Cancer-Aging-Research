import re
import pandas as pd

# Read the file
file_path = '/Users/yashagarwal/Downloads/GSE134379_series_matrix-1.txt'

# Read the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists for columns
sample_titles = []
brain_regions = []
ages = []
diagnoses = []

# Regular expressions to match patterns
brain_region_pattern = re.compile(r'brain region:\s*(\w+)')
age_pattern = re.compile(r'age:\s*(\d+)')
diagnosis_pattern = re.compile(r'diagnosis:\s*([\w\s]+)')

# Parsing logic to fill in sample titles, brain regions, ages, and diagnoses
for line in lines:
    if line.startswith('!Sample_title'):
        titles = line.split('\t')
        sample_titles = [title.strip() for title in titles[1:]]  # Skip the first entry since it's the header
        # Initialize the brain regions, ages, and diagnoses lists with the same length as sample_titles
        brain_regions = [''] * len(sample_titles)
        ages = [''] * len(sample_titles)
        diagnoses = [''] * len(sample_titles)
    if line.startswith('!Sample_characteristics_ch1'):
        characteristics = line.split('\t')
        for i, characteristic in enumerate(characteristics[1:], 1):  # Skip the first entry since it's the header
            brain_region_match = brain_region_pattern.search(characteristic)
            age_match = age_pattern.search(characteristic)
            diagnosis_match = diagnosis_pattern.search(characteristic)
            if brain_region_match:
                brain_regions[i-1] = brain_region_match.group(1)  # Adjust index since titles start from 0
            if age_match:
                ages[i-1] = age_match.group(1)  # Adjust index since titles start from 0
            if diagnosis_match:
                diagnoses[i-1] = diagnosis_match.group(1)  # Adjust index since titles start from 0

# Combine the columns into a structured format
organized_data = list(zip(sample_titles, brain_regions, ages, diagnoses))

df = pd.DataFrame(organized_data, columns=['Sample Title', 'Brain Region', 'Age', 'Diagnosis'])

print(df)
