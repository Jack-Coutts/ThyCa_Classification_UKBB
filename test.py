
import argparse


def sum_nums():

    parser = argparse.ArgumentParser(description='add numbers')  # Initialise parser
    parser.add_argument('--num1', help='number you want added', required=True)
    parser.add_argument('--num2', help='Second num.', required=True)
    args = parser.parse_args()

    return float(args.num1) + float(args.num2)


sm = sum_nums()

print(sm)
