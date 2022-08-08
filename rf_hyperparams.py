import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearnex import patch_sklearn  # Speeds up sklearn with intel patch
patch_sklearn()  # Activate patch - changes sklearn imports below

from timeit import default_timer as timer # Time how long commands take
from sklearn.model_selection import train_test_split, StratifiedKFold  # test_train split, cross-validation

from sklearn.experimental import enable_iterative_imputer  # Iterative imputer experimental so need to enable it
from sklearn.impute import IterativeImputer  # Once enabled iterative imputer can be imported

from sklearn.linear_model import RidgeClassifier, BayesianRidge  # Imputation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder  # Normalisation & Encoding
from imblearn.under_sampling import TomekLinks, RandomUnderSampler  # Undersampling
from imblearn.over_sampling import SMOTENC  # Oversampling
from sklearn.feature_selection import RFE, RFECV  # Recursive feature elimination - feature selection
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # RFE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as imbpipeline

from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer


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


def rf_hyperparams():

    parser = argparse.ArgumentParser(description='Random forest hyperparameter tuning.')  # Initialise parser
    parser.add_argument('--random_state', default=0, type=int)
    parser.add_argument('--iterations', default=50, type=int)
    parser.add_argument('--threads', default=-1, type=int)
    args = parser.parse_args()

    random_state = args.random_state
    n_jobs = args.threads

    # Read in data
    df = pd.read_csv('/data/home/bt211037/dissertation/supervised_ML_data.tsv', sep='\t', header=0, index_col=0)

    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'thyroid_cancer'], df['thyroid_cancer'],
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=random_state,
                                                        stratify=df['thyroid_cancer'])

    # Determine feature types
    cat, con = cat_con_cols(X_train)  # Get the column names of the continuous and nominal data
    X_train[cat] = X_train[cat].astype('Int64')  # Convert categorical cols values from floats to integers - train
    X_test[cat] = X_test[cat].astype('Int64')  # Convert categorical cols values from floats to integers - test

    # Separate categorical and Continuous features in dataframe
    X_train = pd.concat([X_train[con], X_train[cat]], axis=1, join="inner")

    # Column indexes for categorical and continuous features
    categorical_indexes = [i for i, x in enumerate(X_train.columns) if x in cat]
    continuous_indexes = [i for i, x in enumerate(X_train.columns) if x in con]

    # Random forest model
    model = ExtraTreesClassifier(n_jobs=n_jobs, random_state=random_state)

    # Tomek Links undersampling
    tl = TomekLinks(sampling_strategy='majority')

    # SMOTE oversampling
    smote = SMOTENC(random_state=random_state,
                    categorical_features=categorical_indexes,
                    sampling_strategy=1)

    # Imputation column transformer to impute the two different data types
    imputer = ColumnTransformer(
        transformers=[
            ('num', IterativeImputer(initial_strategy='median',
                                     max_iter=5,
                                     random_state=random_state),
             continuous_indexes),

            ('cat', IterativeImputer(estimator=RidgeClassifier(),
                                     initial_strategy='most_frequent',
                                     max_iter=10,
                                     random_state=random_state),
             categorical_indexes)

        ])

    # RFE
    rfe = RFE(model, step=25)

    # Pipeline
    pipeline = imbpipeline(steps=[('imputer', imputer),
                                  ('tomek', tl),
                                  ('smotenc', smote),
                                  ('rfe', rfe),
                                  ('model', model)])

    # Parameters to search
    search_grid = {'rfe__n_features_to_select': range(20, 380, 25),
                   'model__n_estimators': [100, 200, 500, 1000],
                   'model__max_features': range(5, 150, 5),
                   'model__max_depth': [5, 10, 15, 20],
                   'model__bootstrap': [True, False]}

    # Undersample majority class from dataset for smaller dataset to tune hyperparams
    rus = RandomUnderSampler(sampling_strategy=0.01, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)


    rf_hyper_search = RandomizedSearchCV(estimator=pipeline,
                                         param_distributions=search_grid,
                                         n_iter=args.iterations,
                                         cv=5,
                                         n_jobs=n_jobs,
                                         verbose=2,
                                         random_state=random_state,
                                         scoring='f1')

    rf_hyper_search.fit(X_res, y_res)

    results = pd.DataFrame(rf_hyper_search.cv_results_)

    results.to_csv('/data/home/bt211037/dissertation/rf_hyperparam_results.tsv', sep='\t')

    return rf_hyper_search


results = rf_hyperparams()

