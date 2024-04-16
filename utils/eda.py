import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

def compare_means_on_col(df: pd.DataFrame, col):
    '''Compare the means of a column with missing data to the means of a column without missing data'''
    has_data = df[df[col].notnull()]
    no_data = df[df[col].isnull()]
    mean_has_data = has_data.describe().loc['mean']
    mean_no_data = no_data.describe().loc['mean']
    return mean_has_data, mean_no_data

def cohen_d(x, y):
    '''Calculate the effect size using Cohen's d formula'''
    nx = len(x); ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )

def test_hazard_on_col(data: pd.DataFrame, target_col):
    '''Test the hazard of missing data on the target column using t-test and Cohen's d'''
    data = data.select_dtypes(include=['number'])
    bonferonni_correction = len(data.columns) - 1
    try:
        # Split data
        data_with_value = data[data[target_col].notnull()]
        data_without_value = data[data[target_col].isnull()]

        # Ensure there's enough data in both subsets
        if data_with_value.empty or data_without_value.empty:
            print(f"Not enough data to perform tests for column '{target_col}'.")
            return

        results = []
        
        # Loop through all columns except the target column
        for col in data.columns:
            if col == target_col:
                continue
            
            # Get the means
            mean_with = data_with_value[col].dropna().mean()
            mean_without = data_without_value[col].dropna().mean()

            # Perform t-test
            t_stat, p_val = ttest_ind(data_with_value[col].dropna(), data_without_value[col].dropna(), equal_var=False, nan_policy='omit')

            #Perform cohen's d
            d = abs(cohen_d(data_with_value[col].dropna(), data_without_value[col].dropna()))

            #Interprete results
            interpretation = d > .5 and p_val < (.05 / bonferonni_correction)
            
            # Collect results
            results.append((col, mean_with, mean_without, t_stat, p_val, d, interpretation))
        
        # Display results
        results_df = pd.DataFrame(results, columns=['Variable', 'Mean With Data', 'Mean Without Data', 'T-Statistic', 'P-Value', "Cohen's d", "Is Significant"])
        return results_df

    except Exception as e:
        print(f"An error occurred: {e}")

def get_outliers(df, col): 
    '''Get the outliers in a column'''
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]