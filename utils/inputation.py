import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def linear_inputation(df, independent_variable, dependent_variable):
    # Create the regression imputer instance
    imputer_lr = IterativeImputer(estimator=LinearRegression())

    # Impute missing values
    df_imputed_lr = pd.DataFrame(imputer_lr.fit_transform(df), columns=[independent_variable, dependent_variable])
    return df_imputed_lr

def knn_inputation(df, independent_variable, dependent_variable):
    # Create the KNN imputer instance
    imputer_knn = KNNImputer(n_neighbors=2)

    # Impute missing values
    df_imputed_knn = pd.DataFrame(imputer_knn.fit_transform(df), columns=[independent_variable, dependent_variable])
    return df_imputed_knn

def random_forest_inputation(df, independent_variable, dependent_variable):
    # Create the Random Forest imputer instance
    imputer_rf = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)

    # Impute missing values
    df_imputed_rf = pd.DataFrame(imputer_rf.fit_transform(df), columns=[independent_variable, dependent_variable])
    return df_imputed_rf