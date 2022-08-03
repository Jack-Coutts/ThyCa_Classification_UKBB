from functions import *
import logging
from pathlib import Path
import argparse


one_hot_labels = ['Weight method|x21_0_0', 'Spirometry method|x23_0_0', 'Sex|x31_0_0',
                   'UK Biobank assessment centre|x54_0_0', 'Birth weight known|x120_0_0',
                   'Type of accommodation lived in|x670_0_0',
                   'Own or rent accommodation lived in|x680_0_0', 'Drive faster than motorway speed limit|x1100_0_0',
                   'Usual side of head for mobile phone use|x1150_0_0', 'Usual side of head for mobile phone use|x1150_0_0',
                   'Morning/evening person (chronotype)|x1180_0_0', 'Nap during day|x1190_0_0', 'Snoring|x1210_0_0',
                   'Daytime dozing / sleeping (narcolepsy)|x1220_0_0', 'Current tobacco smoking|x1239_0_0',
                   'Past tobacco smoking|x1249_0_0', 'Major dietary changes in the last 5 years|x1538_0_0',
                   'Variation in diet|x1548_0_0',  'Alcohol usually taken with meals|x1618_0_0',
                   'Alcohol intake versus 10 years previously|x1628_0_0', 'Skin colour|x1717_0_0',
                   'Ease of skin tanning|x1727_0_0', 'Hair colour (natural before greying)|x1747_0_0',
                   'Facial ageing|x1757_0_0', 'Father still alive|x1797_0_0', 'Mother still alive|x1835_0_0',
                   'Mood swings|x1920_0_0', 'Miserableness|x1930_0_0', 'Irritability|x1940_0_0',
                   'Sensitivity / hurt feelings|x1950_0_0', 'Fed-up feelings|x1960_0_0', 'Nervous feelings|x1970_0_0',
                   'Worrier / anxious feelings|x1980_0_0', "Tense / 'highly strung'|x1990_0_0",
                   'Worry too long after embarrassment|x2000_0_0', "Suffer from 'nerves'|x2010_0_0",
                   'Loneliness isolation|x2020_0_0', 'Guilty feelings|x2030_0_0', 'Risk taking|x2040_0_0',
                   'Seen doctor (GP) for nerves anxiety tension or depression|x2090_0_0',
                   'Seen a psychiatrist for nerves anxiety tension or depression|x2100_0_0',
                   'Able to confide|x2110_0_0',
                   'Answered sexual history questions|x2129_0_0',
                   'Ever had same-sex intercourse|x2159_0_0', 'Long-standing illness disability or infirmity|x2188_0_0',
                   'Wears glasses or contact lenses|x2207_0_0', 'Other eye problems|x2227_0_0',
                   'Plays computer games|x2237_0_0', 'Hearing difficulty/problems|x2247_0_0',
                   'Hearing difficulty/problems with background noise|x2257_0_0', 'Use of sun/uv protection|x2267_0_0',
                   'Weight change compared with 1 year ago|x2306_0_0',
                   'Wheeze or whistling in the chest in last year|x2316_0_0',
                   'Chest pain or discomfort|x2335_0_0',
                   'Ever had bowel cancer screening|x2345_0_0',
                   'Diabetes diagnosed by doctor|x2443_0_0',
                   'Cancer diagnosed by doctor|x2453_0_0',
                   'Fractured/broken bones in last 5 years|x2463_0_0',
                   'Other serious medical condition/disability diagnosed by doctor|x2473_0_0',
                   'Taking other prescription medications|x2492_0_0',
                   'Pace-maker|x3079_0_0', 'Contra-indications for spirometry|x3088_0_0',
                   'Caffeine drink within last hour|x3089_0_0', 'Used an inhaler for chest within last hour|x3090_0_0',
                   'Method of measuring blood pressure|x4081_0_0', 'Qualifications|x6138_0_0',
                   'Gas or solid-fuel cooking/heating|x6139_0_0',
                   'How are people in household related to participant|x6141_0_0',
                   'Current employment status|x6142_0_0',
                   'Never eat eggs dairy wheat sugar|x6144_0_0',
                   'Illness injury bereavement stress in last 2 years|x6145_0_0',
                   'Attendance/disability/mobility allowance|x6146_0_0',
                   'Mouth/teeth dental problems|x6149_0_0',
                   'Medication for pain relief constipation heartburn|x6154_0_0',
                   'Vitamin and mineral supplements|x6155_0_0', 'Pain type(s) experienced in last month|x6159_0_0',
                   'Leisure/social activities|x6160_0_0',
                   'Types of transport used (excluding work)|x6162_0_0',
                   'Types of physical activity in last 4 weeks|x6164_0_0',
                   'Mineral and other dietary supplements|x6179_0_0',
                   'Illnesses of father|x20107_0_0',
                   'Illnesses of mother|x20110_0_0',
                   'Illnesses of siblings|x20111_0_0',
                   'Smoking status|x20116_0_0',
                   'Alcohol drinker status|x20117_0_0',
                   'Home area population density - urban or rural|x20118_0_0',
                   'Ever smoked|x20160_0_0',  'Spirometry QC measure|x20255_0_0', 'Genetic sex|x22001_0_0',
                   'Genetic kinship to other participants|x22021_0_0',
                   'IPAQ activity group|x22032_0_0', 'Summed days activity|x22033_0_0',
                   'Above moderate/vigorous recommendation|x22035_0_0',
                   'Above moderate/vigorous/walking recommendation|x22036_0_0',
                   'Close to major road|x24014_0_0', 'medication_cbi' ]


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
