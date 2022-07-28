from functions import *
import logging
from pathlib import Path

# Store current filename
current_filename = Path(__file__).stem

# Logger configuration
logging.basicConfig(filename=f'{current_filename}.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG, datefmt='%d-%m-%Y %H:%M:%S')


# Preprocessing function
def preprocessor(data_path, one_hot_encoding=None, random_state=42):

    logging.info(f'Preprocessing function called.')

    X_train, X_test, y_train, y_test = split_tsv(data_path, random_state=random_state)  # Test-Train split
    logging.info(f'Test-train split complete.')

    cat, con = cat_con_cols(X_train)  # Get the column names of the continuous and nominal data
    X_train[cat] = X_train[cat].astype('Int64')  # Convert categorical cols values from floats to integers - train
    X_test[cat] = X_test[cat].astype('Int64')  # Convert categorical cols values from floats to integers - test
    logging.info(f'Categorical columns identified & converted to integer.')

    X_train = minmax_scaling(X_train, con)  # Normalisation
    logging.info(f'Data normalisation complete.')

    X_train = categorical_imputer(X_train, cat, random_state=random_state)  # Imputation
    logging.info(f'X_train categorical imputation complete.')
    X_train = continuous_data(X_train, con, random_state=random_state)  # Imputation
    logging.info(f'X_train continuous imputation complete.')

    X_test = minmax_scaling(X_test, con)  # Normalisation
    X_test = categorical_imputer(X_test, cat, random_state=random_state)  # Imputation
    X_test = continuous_data(X_test, con, random_state=random_state)  # Imputation
    logging.info(f'X_test imputation complete.')

    if one_hot_encoding is not None:
        X_train = feature_encoding(X_train, Onehot=one_hot_encoding)  # One hot encoding
        X_test = feature_encoding(X_test, Onehot=one_hot_encoding)  # One hot encoding
        logging.info(f'One hot encoding complete.')

    X_train, y_train = st_sampling(X_train, y_train, 0.2, random_state=random_state)  # Over/undersampling
    X_test, y_test = st_sampling(X_test, y_test, 0.2, random_state=random_state)  # Over/undersampling
    logging.info(f'Sampling complete.')

    # Save imputed dataframes
    X_train.to_csv('/data/home/bt211037/dissertation/preprocessed_data/X_train.tsv', sep='\t')
    X_test.to_csv('/data/home/bt211037/dissertation/preprocessed_data/X_test.tsv', sep='\t')
    y_train.to_csv('/data/home/bt211037/dissertation/preprocessed_data/y_train.tsv', sep='\t')
    y_test.to_csv('/data/home/bt211037/dissertation/preprocessed_data/y_test.tsv', sep='\t')
    logging.info(f'Files saved, preprocessing complete.')

    return f'Preprocessing complete.'



