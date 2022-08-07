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
from imblearn.under_sampling import TomekLinks  # Undersampling
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
    parser.add_argument('--K_folds', help='Num of stratified cross-validation folds for search', type=int, default=5)
    parser.add_argument('--random_state', default=0, type=int)
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

    # Simple imputation
    simp_imputer = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(missing_values=np.nan, strategy='mean'),
             continuous_indexes),

            ('cat', SimpleImputer(missing_values=np.nan, strategy='constant',
                                  fill_value=4444),
             categorical_indexes)])

    # Pipeline
    pipeline = imbpipeline(steps=[('imputer', simp_imputer),
                                  ('tomek', tl),
                                  ('smotenc', smote),
                                  ('model', model)])

    search_grid = {'model__n_estimators': [500],
                   'model__max_features': [15, 10, 5 ,50],
                   'model__max_depth': [5, 10, 15],
                   'model__bootstrap': [True, False]}

    rf_hyper_search = RandomizedSearchCV(estimator=pipeline,
                                         param_distributions=search_grid,
                                         n_iter=20,
                                         cv=3,
                                         n_jobs=10,
                                         verbose=2,
                                         random_state=random_state)

    rf_hyper_search.fit(X_train, y_train)

    results = pd.DataFrame(pipeline.named_steps['rfe'].cv_results_)

    results.to_csv('/data/home/bt211037/dissertation/rf_hyperparam_results.tsv', sep='\t')

    return rf_hyper_search



