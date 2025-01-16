from rpy2.robjects import r

# Set R options to suppress warnings
r('options(warn = -1)')

# Load your libraries afterwards
r('.libPaths("/home-nfs2/local/VANDERBILT/samirj/R/x86_64-pc-linux-gnu-library/3.6")')
r('library(gamlss)')
r('library(mgcv)')  

from rpy2.robjects import r
r('.libPaths("/home-nfs2/local/VANDERBILT/samirj/R/x86_64-pc-linux-gnu-library/3.6")')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects import r
from scipy.stats import sem
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter, globalenv
pandas2ri.activate()
gamlss = importr("gamlss")
base = importr("base")
utils = importr("utils")


# Load the dataset
data_file = "/home/local/VANDERBILT/samirj/3rdyearproj/lgdata/COMBINED_FILTERED_QA.csv"
data = pd.read_csv(data_file, low_memory = False)

# Prompt user for tract name and feature
tract = input("Enter the tract name (e.g., AF, UF, SLF): ").strip()
feature = input("Enter the feature to analyze (e.g., volume, FA): ").strip()

# Validate that the necessary columns exist
columns_to_keep = [
    "scan", "subject", "session", "dataset", "age", "sex",
    f"{tract}_left-{feature}", f"{tract}_right-{feature}"
]

for column in columns_to_keep[6:]:
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset. Please check the tract and feature names.")

data = data[columns_to_keep]
data['session'] = data['session'].astype(str)
data.dropna(inplace=True)

print(data.columns)

# Separate data by sex
male_data = data[data["sex"] == 1]
female_data = data[data["sex"] == 0]


def process_data_and_fit_gamlss(data, tract, feature):
    # Compute asymmetry: (Left - Right) / (Left + Right)
    asym_col_name = f"{tract}_asymmetry_{feature}"
    data = data.copy()
    data[asym_col_name] = (
        data[f"{tract}_left-{feature}"] - data[f"{tract}_right-{feature}"]
    ) / (
        data[f"{tract}_left-{feature}"] + data[f"{tract}_right-{feature}"]
    )
    
    # Convert the pandas DataFrame to an R dataframe
    with localconverter(pandas2ri.converter) as cv:
        r_data = pandas2ri.py2rpy(data)

    globalenv["r_data"] = r_data
    
    # Fit the GAMLSS model with a non-linear trend
    gamlss_formula = r(f'{asym_col_name} ~ age')
    gamlss_family = r('NO')  
    gamlss_model = gamlss.gamlss(formula=gamlss_formula, family=gamlss_family, data=r_data)
    
    # Return the fitted model, asymmetry column name, and modified data
    return gamlss_model, asym_col_name, data

# Fit GAMLSS models - remove the 'Male'/'Female' string from the call
male_model, male_asym_col_name, male_data = process_data_and_fit_gamlss(male_data, tract, feature)
female_model, female_asym_col_name, female_data = process_data_and_fit_gamlss(female_data, tract, feature)



# Function to calculate percentiles and confidence intervals
def percentiles_and_cis(data, asym_col_name):
    # Calculate the percentiles
    percentiles = data.groupby('age')[asym_col_name].quantile([0.25, 0.5, 0.75]).unstack(level=1)
    # Calculate the median (50th percentile)
    median = percentiles[0.5]
    
    # Calculate the Standard Error of the Mean (SEM) for the age groups
    age_groups = data.groupby('age')[asym_col_name]
    sem_values = age_groups.apply(lambda x: sem(x) if len(x) > 1 else np.nan).rename('SEM')
    
    # Calculate the 95% confidence intervals for the median
    ci_lower = median.sub(sem_values.mul(1.96), fill_value=np.nan)
    ci_upper = median.add(sem_values.mul(1.96), fill_value=np.nan)
    
    return percentiles, ci_lower, ci_upper, median.index, median.values

# Calculate percentiles and confidence intervals for male and female
male_percentiles, male_ci_lower, male_ci_upper, male_ages_unique, male_median = percentiles_and_cis(male_data, male_asym_col_name)
female_percentiles, female_ci_lower, female_ci_upper, female_ages_unique, female_median = percentiles_and_cis(female_data, female_asym_col_name)

with localconverter(robjects.default_converter + pandas2ri.converter):
    male_r_age_dataframe = robjects.conversion.py2rpy(pd.DataFrame({'age': male_data['age'].unique()}))
    female_r_age_dataframe = robjects.conversion.py2rpy(pd.DataFrame({'age': female_data['age'].unique()}))

def get_gamlss_fitted_values(gamlss_model, r_age_dataframe):
    # Use the R predict function from the gamlss package namespace
    predict_func = robjects.r['predict']
    # Make the prediction using the predict func and the R dataframe
    predictions = predict_func(gamlss_model, newdata=r_age_dataframe, type='response')
    
    # Directly return the predictions without conversion
    return predictions

# Obtain the predict function from R's gamlss package
predict_func = robjects.r['predict']

# Make predictions for male and female data
male_fitted_array = get_gamlss_fitted_values(male_model, male_r_age_dataframe)
female_fitted_array = get_gamlss_fitted_values(female_model, female_r_age_dataframe)

# Convert the numpy arrays to pandas Series if needed
male_fitted = pd.Series(male_fitted_array)
female_fitted = pd.Series(female_fitted_array)



# Make sure that the fitted values and ages are numpy arrays (if not, convert them)
# Sort and align the ages and fitted values for plotting
male_ages_sorted = np.sort(male_data['age'].unique())
female_ages_sorted = np.sort(female_data['age'].unique())

# Fit GAMLSS models
male_model, male_asym_col_name, male_data = process_data_and_fit_gamlss(male_data, tract, feature)
female_model, female_asym_col_name, female_data = process_data_and_fit_gamlss(female_data, tract, feature)

# Assuming age_data is a dataframe with a single column 'age' where the column named 'age' consists of the
# ages you want to predict for. Example below assumes 'age' is already a column in male_data and female_data.
male_ages = robjects.FloatVector(male_data['age'].unique())
female_ages = robjects.FloatVector(female_data['age'].unique())

# Obtain the fitted values
male_fitted = get_gamlss_fitted_values(male_model, male_r_age_dataframe)
female_fitted = get_gamlss_fitted_values(female_model, female_r_age_dataframe)


# Plotting the results with median and confidence interval
plt.figure(figsize=(14, 7))

# Male plot
plt.subplot(1, 2, 1)
plt.scatter(male_data['age'], male_data[male_asym_col_name], label="Male Data", color="blue", alpha=0.6)
plt.plot(male_ages_unique, male_median, label="Median", color="green", linestyle='--')
plt.fill_between(male_ages_unique, male_ci_lower, male_ci_upper, color='gray', alpha=0.2, label="95% CI of Median")
plt.plot(male_ages, male_fitted, label="GAMLSS Fitted Trend", color="black")
plt.title(f"Male {feature} Asymmetry in the {tract} Tract")
plt.xlabel("Age")
plt.ylabel(f"Asymmetry Index")
plt.legend()

# Female plot
plt.subplot(1, 2, 2)
plt.scatter(female_data['age'], female_data[female_asym_col_name], label="Female Data", color="red", alpha=0.6)
plt.plot(female_ages_unique, female_median, label="Median", color="green", linestyle='--')
plt.fill_between(female_ages_unique, female_ci_lower, female_ci_upper, color='gray', alpha=0.2, label="95% CI of Median")
plt.plot(female_ages, female_fitted, label="GAMLSS Fitted Trend", color="black")
plt.title(f"Female {feature} Asymmetry in the {tract} Tract")
plt.xlabel("Age")
plt.ylabel(f"Asymmetry Index")
plt.legend()

plt.tight_layout()
plt.show()