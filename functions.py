# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import random

from timeit import default_timer as timer # Time how long commands take
from sklearn.model_selection import train_test_split, StratifiedKFold  # test_train split, cross-validation
from sklearn.experimental import enable_iterative_imputer  # Iterative imputer experimental so need to enable it
from sklearn.impute import IterativeImputer  # Once enabled iterative imputer can be imported
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier  # Imputation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # Normalisation & Encoding
from imblearn.combine import SMOTETomek  # Sampling
from sklearn.feature_selection import RFE, RFECV  # Recursive feature elimination - feature selection
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# Stopwatch to profile function runtimes
class Stopwatch:

    # Initiate constructor
    def __init__(self):
        self.start = timer()
        self.end = None
        self.runtime = None

    # Stop stopwatch
    def stop(self):
        self.end = timer()
        self.runtime = self.end - self.start
        return self.runtime


# Produce test/train split from csv/tsv file
def split_tsv(file_path, random_state, sep='\t', test_size=0.2):

    df = pd.read_csv(file_path, sep=sep, header=0, index_col=0)
    # y = df['thyroid_cancer']
    # X = df.loc[:, df.columns != 'thyroid_cancer']
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'thyroid_cancer'],
                                                        df['thyroid_cancer'],
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


# Find which columns are categorical and which continuous
def cat_con_cols(df):
    columns = [list(df[i]) for i in df]  # Nested list of column values
    num_unique_vals = [len(set([i for i in a if pd.notna(i)])) for a in columns]  # Num of unique values in a column

    categorical_indexes = [i for i, v in enumerate(num_unique_vals) if v <= 100
                           and set([a % 1 for a in df[df.columns[i]].dropna()]) == set([0])]

    continuous_indexes = [i for i, v in enumerate(num_unique_vals) if v > 100 or
                          set([a % 1 for a in df[df.columns[i]].dropna()]) != set([0])]

    cat = list(df.columns[categorical_indexes])
    con = list(df.columns[continuous_indexes])
    return cat, con


# Extratrees imputer - equivalent to missforest
def extra_trees_imputer(dataframe, cat, con, random_state):

    cat_imputer = IterativeImputer(estimator=ExtraTreesClassifier(n_estimators=10,n_jobs=-1, verbose=0),
                                   initial_strategy='most_frequent', max_iter=5, random_state=random_state, verbose=2)

    imputed_cat = cat_imputer.fit_transform(dataframe[cat])

    con_imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, n_jobs=-1, verbose=0),
                                   initial_strategy='mean', max_iter=5, random_state=random_state, verbose=2)

    imputed_con = con_imputer.fit_transform(dataframe[con])

    dataframe.loc[:, cat] = imputed_cat
    dataframe.loc[:, con] = imputed_con

    return dataframe


# Normalised feature scaling
def minmax_scaling(df, continuous_data):

    outdata = df
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[continuous_data])
    outdata[continuous_data] = scaled

    return outdata


# Feature encoding
def feature_encoding(dataframe, ordinal=None, Onehot=None):
    if ordinal is not None:
        ordinal_encoder = OrdinalEncoder()
        ordinal_data = ordinal_encoder.fit_transform(dataframe[ordinal])
        dataframe[ordinal] = ordinal_data

    if Onehot is not None:

        encode_targets = []  # Remove binary columns
        for item in Onehot:
            col = dataframe[item]
            unique = len(set(col))
            if unique > 2:
                encode_targets.append(item)

        onehot = OneHotEncoder()
        onehot_data = onehot.fit_transform(dataframe[encode_targets]).toarray()  # Create new cols
        one_hot_names = onehot.get_feature_names_out(encode_targets)  # Get new col names
        onehot_df = pd.DataFrame(onehot_data, columns=one_hot_names, index=dataframe.index)  # Create df of new cols
        dataframe = dataframe.drop(axis=1, labels=Onehot)  # Drop original columns
        dataframe = dataframe.join(onehot_df)

    return dataframe


# Over and undersampling with SMOTE-Tomek links
def st_sampling(X_df, y_df, ratio, random_state):
    smt = SMOTETomek(random_state=random_state, sampling_strategy=ratio)
    X_res, y_res = smt.fit_resample(X_df, y_df)
    return X_res, y_res