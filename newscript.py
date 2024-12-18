from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


# Activate the automatic conversion of rpy2 objects to pandas objects
pandas2ri.activate()

# Suppress R warnings
r('options(warn=-1)')

# Load required R libraries
gamlss = importr('gamlss')
r('library(mgcv)')  # Load the mgcv package here

# Load the dataset
data_file = "/home/local/VANDERBILT/samirj/3rdyearproj/lgdata/COMBINED_FILTERED_QA.csv"
data = pd.read_csv(data_file, low_memory=False)

# Define a function to process the data and fit a GAMLSS model
def process_data_and_fit_gamlss(data, tract, feature):
    # Compute asymmetry: (Left - Right) / (Left + Right)
    asym_col_name = f"{tract}_asymmetry_{feature}"
    data = data.copy()
    data[asym_col_name] = (
        data[f"{tract}_left-{feature}"] - data[f"{tract}_right-{feature}"]
    ) / (
        data[f"{tract}_left-{feature}"] + data[f"{tract}_right-{feature}"]
    )

    # Convert pandas DataFrame to R dataframe
    with localconverter(pandas2ri.converter) as cv:
        r_data = pandas2ri.py2rpy(data)

    # Fit the GAMLSS model using penalized B-splines for non-linear trends
    gamlss_formula = r(f'{asym_col_name} ~ pb(age)')
    gamlss_family = r('NO')
    gamlss_model = r['gamlss'](formula=gamlss_formula, family=gamlss_family, data=r_data)

    return gamlss_model, asym_col_name, data


# Function to extract GAMLSS fitted and predicted values
def get_gamlss_fitted_values(gamlss_model, ages):
    new_data = robjects.DataFrame({'age': robjects.FloatVector(ages)})
    predict_func = r['predict']
    predictions = predict_func(gamlss_model, newdata=new_data, type='response')
    return np.array(predictions)

# Separate your data by sex and define tract and feature variables
tract = "AF"
feature = "volume"

# Define columns to keep, drop NAs, and drop duplicates
columns_to_keep = ['scan', 'subject', 'session', 'dataset', 'age', 'sex', f'{tract}_left-{feature}', f'{tract}_right-{feature}']
data = data[columns_to_keep]
data = data.dropna()
data = data.drop_duplicates()

# Get unique sorted ages
unique_ages = data['age'].unique()
unique_ages.sort()

# Split the data by sex
male_data = data[data['sex'] == 1]
female_data = data[data['sex'] == 0]

# Fit the models
male_model, male_asym_col_name = process_data_and_fit_gamlss(male_data, tract, feature)
female_model, female_asym_col_name = process_data_and_fit_gamlss(female_data, tract, feature)

# Get the fitted values
male_fitted_values = get_gamlss_fitted_values(male_model, unique_ages)
female_fitted_values = get_gamlss_fitted_values(female_model, unique_ages)

# Plotting
plt.subplot(1, 2, 1)
plt.scatter(male_data['age'], male_data[male_asym_col_name], label="Male Data", color="blue", alpha=0.6)
plt.plot(unique_ages, male_fitted_values, label="GAMLSS Fitted Trend", color="black")
plt.title(f"Male {feature} Asymmetry in the {tract} Tract")
plt.xlabel('Age')
plt.ylabel(f'{feature} Asymmetry ({tract})')
plt.legend()

# Female plot
plt.subplot(1, 2, 2)
plt.scatter(female_data['age'], female_data[female_asym_col_name], label="Female Data", color="red", alpha=0.6)
plt.plot(unique_ages, female_fitted_values, label="GAMLSS Fitted Trend", color="black")
plt.title(f"Female {feature} Asymmetry in the {tract} Tract")
plt.xlabel('Age')
plt.ylabel(f'{feature} Asymmetry ({tract})')
plt.legend()

plt.tight_layout()
plt.show()