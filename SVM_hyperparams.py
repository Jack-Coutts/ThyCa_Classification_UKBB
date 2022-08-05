
import logging
import argparse
import pandas as pd
import numpy as np

from sklearnex import patch_sklearn  # Speeds up sklearn with intel patch
patch_sklearn()  # Activate patch - changes sklearn imports below

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.experimental import enable_iterative_imputer  # Iterative imputer experimental so need to enable it
from sklearn.impute import IterativeImputer  # Once enabled iterative imputer can be imported
from sklearn.linear_model import RidgeClassifier, BayesianRidge  # Imputation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # Normalisation & Encoding
from sklearn.feature_selection import RFE, RFECV  # Recursive feature elimination - feature selection
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import TomekLinks  # Undersampling
from imblearn.over_sampling import SMOTENC  # Oversampling
from imblearn.pipeline import Pipeline as imbpipeline




def svm_search():

    """ Function to run a support vector machine hyperparameter grid search on UK Biobank data for thyroid cancer
        classification. """

    # Establish command-line interface
    parser = argparse.ArgumentParser(description='SVM hyperparameter grid search')  # Initialise parser
    parser.add_argument('--dataset', help='TSV file containing participant data', required=True)
    parser.add_argument('--K_folds', help='Num of stratified cross-validation folds for search', type=int, default=5)
    parser.add_argument('--random_state', default=0, type=int)
    args = parser.parse_args()

    # Data cleaning
    df = pd.read_csv(args.dataset, sep='\t', header=0, index_col=0)  # Read in data








    pass