import pandas as pd
from sklearn.model_selection import train_test_split

# Manual feature selection from tsv


# Produce test/train split from csv/tsv file
def split_tsv(file_path, sep='\t', test_size=0.2):

    df = pd.read_csv(file_path, sep=sep, header=0)
    y = df['thyroid_cancer']
    X = df.loc[:, df.columns != 'thyroid_cancer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    return X_train, X_test, y_train, y_test






# Feature encoding


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





