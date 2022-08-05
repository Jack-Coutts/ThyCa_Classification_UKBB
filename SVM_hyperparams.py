
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


def cat_con_cols(df):

    """ Determine which columns contain categorical data and which contain continuous data in a dataframe """

    columns = [list(df[i]) for i in df]  # Nested list of column values
    num_unique_vals = [len(set([i for i in a if pd.notna(i)])) for a in columns]  # Num of unique values in a column

    categorical_indexes = [i for i, v in enumerate(num_unique_vals) if v <= 100
                           and set([a % 1 for a in df[df.columns[i]].dropna()]) == {0}]

    continuous_indexes = [i for i, v in enumerate(num_unique_vals) if v > 100 or
                          set([a % 1 for a in df[df.columns[i]].dropna()]) != {0}]

    cat = list(df.columns[categorical_indexes])
    con = list(df.columns[continuous_indexes])
    return cat, con


def select_onehot(X_train):

    """ Determine the column indexes of features that need to be onehot encoded. Which features need to be one hot
        encoded was decided manually.
    """

    # Columns requiring one hot encoding
    onehot = ['Weight method|x21_0_0', 'Spirometry method|x23_0_0', 'UK Biobank assessment centre|x54_0_0',
              'Birth weight known|x120_0_0', 'Type of accommodation lived in|x670_0_0',
              'Own or rent accommodation lived in|x680_0_0', 'Drive faster than motorway speed limit|x1100_0_0',
              'Usual side of head for mobile phone use|x1150_0_0', 'Usual side of head for mobile phone use|x1150_0_0',
              'Morning/evening person (chronotype)|x1180_0_0', 'Nap during day|x1190_0_0', 'Snoring|x1210_0_0',
              'Daytime dozing / sleeping (narcolepsy)|x1220_0_0', 'Current tobacco smoking|x1239_0_0',
              'Past tobacco smoking|x1249_0_0', 'Major dietary changes in the last 5 years|x1538_0_0',
              'Variation in diet|x1548_0_0', 'Alcohol usually taken with meals|x1618_0_0',
              'Alcohol intake versus 10 years previously|x1628_0_0', 'Skin colour|x1717_0_0',
              'Ease of skin tanning|x1727_0_0', 'Hair colour (natural before greying)|x1747_0_0',
              'Facial ageing|x1757_0_0', 'Father still alive|x1797_0_0', 'Mother still alive|x1835_0_0',
              'Mood swings|x1920_0_0', 'Miserableness|x1930_0_0', 'Irritability|x1940_0_0',
              'Sensitivity / hurt feelings|x1950_0_0', 'Fed-up feelings|x1960_0_0', 'Nervous feelings|x1970_0_0',
              'Worrier / anxious feelings|x1980_0_0', "Tense / 'highly strung'|x1990_0_0",
              'Worry too long after embarrassment|x2000_0_0', "Suffer from 'nerves'|x2010_0_0",
              'Loneliness isolation|x2020_0_0', 'Guilty feelings|x2030_0_0', 'Risk taking|x2040_0_0',
              'Seen doctor (GP) for nerves anxiety tension or depression|x2090_0_0',
              'Seen a psychiatrist for nerves anxiety tension or depression|x2100_0_0', 'Able to confide|x2110_0_0',
              'Answered sexual history questions|x2129_0_0', 'Ever had same-sex intercourse|x2159_0_0',
              'Long-standing illness disability or infirmity|x2188_0_0', 'Wears glasses or contact lenses|x2207_0_0',
              'Other eye problems|x2227_0_0', 'Plays computer games|x2237_0_0', 'Hearing difficulty/problems|x2247_0_0',
              'Hearing difficulty/problems with background noise|x2257_0_0', 'Use of sun/uv protection|x2267_0_0',
              'Weight change compared with 1 year ago|x2306_0_0',
              'Wheeze or whistling in the chest in last year|x2316_0_0',
              'Chest pain or discomfort|x2335_0_0', 'Ever had bowel cancer screening|x2345_0_0',
              'Diabetes diagnosed by doctor|x2443_0_0', 'Cancer diagnosed by doctor|x2453_0_0',
              'Fractured/broken bones in last 5 years|x2463_0_0',
              'Other serious medical condition/disability diagnosed by doctor|x2473_0_0',
              'Taking other prescription medications|x2492_0_0', 'Pace-maker|x3079_0_0',
              'Contra-indications for spirometry|x3088_0_0', 'Caffeine drink within last hour|x3089_0_0',
              'Used an inhaler for chest within last hour|x3090_0_0', 'Method of measuring blood pressure|x4081_0_0',
              'Qualifications|x6138_0_0', 'Gas or solid-fuel cooking/heating|x6139_0_0',
              'How are people in household related to participant|x6141_0_0', 'Current employment status|x6142_0_0',
              'Never eat eggs dairy wheat sugar|x6144_0_0',
              'Illness injury bereavement stress in last 2 years|x6145_0_0',
              'Attendance/disability/mobility allowance|x6146_0_0', 'Mouth/teeth dental problems|x6149_0_0',
              'Medication for pain relief constipation heartburn|x6154_0_0',
              'Vitamin and mineral supplements|x6155_0_0',
              'Pain type(s) experienced in last month|x6159_0_0', 'Leisure/social activities|x6160_0_0',
              'Types of transport used (excluding work)|x6162_0_0',
              'Types of physical activity in last 4 weeks|x6164_0_0',
              'Mineral and other dietary supplements|x6179_0_0', 'Illnesses of father|x20107_0_0',
              'Illnesses of mother|x20110_0_0', 'Illnesses of siblings|x20111_0_0', 'Smoking status|x20116_0_0',
              'Alcohol drinker status|x20117_0_0', 'Home area population density - urban or rural|x20118_0_0',
              'Spirometry QC measure|x20255_0_0', 'Genetic sex|x22001_0_0',
              'Genetic kinship to other participants|x22021_0_0', 'IPAQ activity group|x22032_0_0',
              'Summed days activity|x22033_0_0', 'Above moderate/vigorous recommendation|x22035_0_0',
              'Above moderate/vigorous/walking recommendation|x22036_0_0', 'Close to major road|x24014_0_0',
              'medication_cbi']

    # Remove any duplicates
    onehot = set(onehot)

    # Get indexes of these columns for one hot encoding
    oh_indexes = [i for i, feat in enumerate(X_train.columns) if feat in onehot]

    return oh_indexes


def svm_search():

    """
    Function to run a support vector machine hyperparameter grid search on UK Biobank data for thyroid cancer
    classification.
    """

    # Establish command-line interface
    parser = argparse.ArgumentParser(description='SVM hyperparameter grid search')  # Initialise parser
    parser.add_argument('--dataset', help='TSV file containing participant data', required=True)
    parser.add_argument('--K_folds', help='Num of stratified cross-validation folds for search', type=int, default=5)
    parser.add_argument('--random_state', default=0, type=int)
    args = parser.parse_args()

    # Read in dataset
    df = pd.read_csv(args.dataset, sep='\t', header=0, index_col=0)

    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'thyroid_cancer'],
                                                        df['thyroid_cancer'],
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=args.random_state)

    # Find categorical and continuous features
    cat, con = cat_con_cols(X_train)

    # Convert categorical columns/features from floats to integers - for test and train data
    X_train[cat] = X_train[cat].astype('Int64')
    X_test[cat] = X_test[cat].astype('Int64')

    # Re-arrange dataframe columns, grouping cat cols and con cols, to aid in later indexing
    X_train = pd.concat([X_train[con], X_train[cat]], axis=1, join="inner")

    # Get the column indexes of features needing one hot encoding
    oh_indexes = select_onehot(X_train)

    # Get the column indexes of categorical and continuous features
    categorical_indexes = [i for i, x in enumerate(X_train.columns) if x in cat]
    continuous_indexes = [i for i, x in enumerate(X_train.columns) if x in con]

    # Classifier model for recursive feature elimination (RFE)
    rfe_model = RidgeClassifier(alpha=1)

    # Cross validation splitting strategy for RFE
    rfe_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    pass