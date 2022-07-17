import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Manual feature selection from tsv


# Produce test/train split from csv/tsv file
def split_tsv(file_path, sep='\t', test_size=0.2):

    df = pd.read_csv(file_path, sep=sep, header=0, index_col=0)
    y = df['thyroid_cancer']
    X = df.loc[:, df.columns != 'thyroid_cancer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    return X_train, X_test, y_train, y_test


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



# Imputation


# Feature scaling


# Undersampling


# Oversampling


# Feature selection & hyperparemeter tuning (grid search)


# Cross validation


# Model training


# Model evaluation


# Save model


# Model explainability





