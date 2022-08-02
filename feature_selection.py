from functions import rfecv
import argparse


def feature_selector():

    """ Set up argparser for command line interface """

    parser = argparse.ArgumentParser(description='Cross-Validated Recursive Feature Elimination')  # Initialise parser
    parser.add_argument('--X_train', help='training data', required=True)
    parser.add_argument('--y_train', help='training target', required=True)
    parser.add_argument('--n_estimators', help='number of estimators in in ExtraTreesClassifier', default=10, type=int)
    parser.add_argument('--n_folds', help='number of coross-validation folds.', default=5, type=int)
    parser.add_argument('--plotfile', help='file path to plot', required=True)

    args = parser.parse_args()

    feature_names = rfecv(args.X_train, args.y_train, args.n_estimators, args.n_folds, args.plotfile)

    return feature_names


names = feature_selector()

with open('/data/home/bt211037/dissertation/preprocessed_data/selected_features.txt', 'w') as f:
    for feature in names:
        f.write(f"{feature}\n")

print(f'Feature selection complete.')