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
def preprocessor():

    logging.info(f'Preprocessing function called.')

    random_state = 42  # Set random state
    # Test-Train split
    X_train, X_test, y_train, y_test = split_tsv('/data/home/bt211037/dissertation/supervised_ML_data.tsv',
                                                 random_state=random_state)

    logging.info(f'Test-train split complete.')

    cat, con = cat_con_cols(X_train)  # Get the column names of the continuous and nominal data
    X_train[cat] = X_train[cat].astype('Int64')  # Convert categorical cols values from floats to integers

    logging.info(f'X_train imputation started.')
    X_train_i4 = extra_trees_imputer(X_train[0:4000], cat, con)



    pass

