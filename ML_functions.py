import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.neighbors._base  # Import required for missforest due to new sklearn version renaming
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest  # Import package for MissForest


# Manual feature selection from tsv


# Stopwatch to time functions
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
def split_tsv(file_path, sep='\t', test_size=0.2):

    df = pd.read_csv(file_path, sep=sep, header=0, index_col=0)
    y = df['thyroid_cancer']
    X = df.loc[:, df.columns != 'thyroid_cancer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    return X_train, X_test, y_train, y_test


# Function to get column names for continuous and non-continuous (categorical) features
def find_cat_or_con_columns(dataframe):

    columns = [list(dataframe[i]) for i in dataframe] # Nested list of column values

    uniques = [len(set([i for i in a if pd.notna(i)])) for a in columns] # Num of unique values in a column

    continuous_indexes = [i for i, c in enumerate(uniques) if c > 50] # Indexes of continuous columns

    categorical_indexes = [i for i, c in enumerate(uniques) if c <= 50] # Indexes of categorical columns

    con_cols = [dataframe.columns[x] for x in continuous_indexes] # List containing continuous columns names

    cat_cols = [dataframe.columns[x] for x in categorical_indexes] # List containing categorical columns names

    return con_cols, cat_cols # return two lists of continuous & categorical column names


# Imputation
def miss_forest_imputation(df):

    con, cat = find_cat_or_con_columns(df)
    cat_indexes = [i for i, x in enumerate(list(df.columns)) if x in cat]

    imputer = MissForest(max_iter=5)  # Initiate missforest
    # Carry out imputation, specified categorical column indexes
    imputed = imputer.fit_transform(df, cat_vars=cat_indexes)
    imputed_df = pd.DataFrame(imputed, columns=df.columns)

    return imputed_df


# Feature encoding - one hot encoding (label encoding already completed)
def feature_encoding(one_hot_cols, df):

    enc = OneHotEncoder()  # Initiate onehotencoder instance
    enc_data = enc.fit_transform(df[one_hot_cols]).toarray()  # One hot encode target cols
    enc_feat = enc.get_feature_names(one_hot_cols)  # Add new column names
    encoded_df = pd.DataFrame(enc_data, columns=enc_feat)  # Convert results to dataframe
    encoded_df = encoded_df.set_index(df.index)  # Ensure new df has same index as original
    df_con = df.drop(one_hot_cols, axis=1)  # remove original columns that were encoded
    dataframe = df_con.join(encoded_df, how='inner')  # Combine the two dataframes

    return dataframe





# Feature scaling


# Undersampling


# Oversampling


# Feature selection & hyperparemeter tuning (grid search)


# Cross validation


# Model training


# Model evaluation


# Save model


# Model explainability





