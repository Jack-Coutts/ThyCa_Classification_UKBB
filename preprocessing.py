from functions import *
import logging
from pathlib import Path
import argparse


# Preprocessing function
def preprocessor():

    """ Set up argparser for command line interface """

    parser = argparse.ArgumentParser(description='Data Preprocessing of UK Biobank Data')  # Initialise parser
    parser.add_argument('--dataset', help='TSV file containing participant data', required=True)
    parser.add_argument('--onehot', help='Should one hot encoding be applied.', default=False)
    parser.add_argument('--random_state', required=True, type=int)
    parser.add_argument('--logfile', help='name of log file', required=True)
    args = parser.parse_args()

    """ Set up logging for function """

    # Store current filename
    current_filename = Path(__file__).stem

    # Logger configuration
    logging.basicConfig(filename=args.logfile, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG, datefmt='%d-%m-%Y %H:%M:%S')

    logging.info(f'Preprocessing function called.')

    """ Conduct preprocessing """

    X_train, X_test, y_train, y_test = split_tsv(args.dataset, args.random_state)  # Test-Train split
    logging.info(f'Test-train split complete.')

    cat, con = cat_con_cols(X_train)  # Get the column names of the continuous and nominal data
    X_train[cat] = X_train[cat].astype('Int64')  # Convert categorical cols values from floats to integers - train
    X_test[cat] = X_test[cat].astype('Int64')  # Convert categorical cols values from floats to integers - test
    logging.info(f'Categorical columns identified & converted to integer.')

    X_train = minmax_scaling(X_train, con)  # Normalisation
    logging.info(f'Data normalisation complete.')

    X_train = categorical_imputer(X_train, cat, args.random_state)  # Imputation
    logging.info(f'X_train categorical imputation complete.')
    X_train = continuous_data(X_train, con, args.random_state)  # Imputation
    logging.info(f'X_train continuous imputation complete.')

    X_test = minmax_scaling(X_test, con)  # Normalisation
    X_test = categorical_imputer(X_test, cat, args.random_state)# Imputation
    X_test = continuous_data(X_test, con, args.random_state)  # Imputation
    logging.info(f'X_test imputation complete.')

    if args.onehot:
        X_train = feature_encoding(X_train, Onehot=one_hot_labels)  # One hot encoding
        X_test = feature_encoding(X_test, Onehot=one_hot_labels)  # One hot encoding
        logging.info(f'One hot encoding complete.')

    X_train, y_train = ou_sampling(X_train, y_train, 1, args.random_state, cat)  # Over/undersampling
    X_test, y_test = ou_sampling(X_test, y_test, 1, args.random_state, cat)  # Over/undersampling
    logging.info(f'Sampling complete.')

    # Save imputed dataframes
    X_train.to_csv('/data/home/bt211037/dissertation/preprocessed_data/X_train.tsv', sep='\t')
    X_test.to_csv('/data/home/bt211037/dissertation/preprocessed_data/X_test.tsv', sep='\t')
    y_train.to_csv('/data/home/bt211037/dissertation/preprocessed_data/y_train.tsv', sep='\t')
    y_test.to_csv('/data/home/bt211037/dissertation/preprocessed_data/y_test.tsv', sep='\t')
    logging.info(f'Files saved, preprocessing complete.')

    return f'Preprocessing complete.'


pre = preprocessor()

print(pre)
