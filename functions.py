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
from sklearn.linear_model import RidgeClassifier, BayesianRidge  # Imputation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder  # Normalisation & Encoding
from imblearn.under_sampling import TomekLinks  # Undersampling
from imblearn.over_sampling import SMOTENC  # Oversampling
from sklearn.feature_selection import RFE, RFECV  # Recursive feature elimination - feature selection
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier  # RFE
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
                           and set([a % 1 for a in df[df.columns[i]].dropna()]) == {0}]

    continuous_indexes = [i for i, v in enumerate(num_unique_vals) if v > 100 or
                          set([a % 1 for a in df[df.columns[i]].dropna()]) != {0}]

    cat = list(df.columns[categorical_indexes])
    con = list(df.columns[continuous_indexes])
    return cat, con


# Categorical imputation using extratreesclassifier
def categorical_imputer(df, cat, random_state):

    cat_imputer = IterativeImputer(estimator=RidgeClassifier(), initial_strategy='most_frequent',
                                   max_iter=10, random_state=random_state, verbose=0)

    imputed_cat = cat_imputer.fit_transform(df[cat])
    df.loc[:, cat] = imputed_cat
    return df


# Continuous imputation using bayesian ridge
def continuous_data(df, con, random_state):

    con_imputer = IterativeImputer(initial_strategy='mean', max_iter=5, random_state=random_state, verbose=0)
    imputed_con = con_imputer.fit_transform(df[con])
    df.loc[:, con] = imputed_con
    return df


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


# SMOTENC and Tomek over and under-sampling
def ou_sampling(X_df, y_df, ratio, random_state, cat):

    indexes = [i for i, x in enumerate(X_df.columns) if x in cat]

    # Tomek links
    tl = TomekLinks(sampling_strategy='majority', n_jobs=-1)
    X_res, y_res = tl.fit_resample(X_df, y_df)

    # SMOTENC - SMOTE for nominal and continuous data
    sm = SMOTENC(random_state=random_state, categorical_features=indexes, sampling_strategy=ratio, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_res, y_res)

    return X_res, y_res


# recursive feature elimination
def rfecv(X_train, y_train, n_estimators, n_folds, plotfile):

    X_train = pd.read_csv(X_train, sep='\t', header=0, index_col=0)
    y_train = pd.read_csv(y_train, sep='\t', header=0, index_col=0)

    model = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
    rfe = RFECV(model, step=1, cv=StratifiedKFold(n_folds), scoring='f1')
    feature_info = rfe.fit(X_train, y_train.values.ravel())

    feat_indexes = [x for x, y in enumerate(feature_info.ranking_) if y == 1]
    feat_names = [X_train.columns[z] for z in feat_indexes]

    # Plot model accuracy against feature num
    plt.figure(figsize=(16, 6))
    plt.xlabel('Total features selected')
    plt.ylabel('Model f1 score')
    plt.plot(range(1, len(feature_info.grid_scores_) + 1), feature_info.grid_scores_)
    plt.savefig(plotfile, bbox_inches='tight')

    return feat_names

