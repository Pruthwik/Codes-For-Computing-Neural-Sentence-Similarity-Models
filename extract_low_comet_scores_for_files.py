"""Extract low COMET scores (threshold is 0.6)."""
from argparse import ArgumentParser
import os
import pandas as pd


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='i', help='Enter the input folder path')
    parser.add_argument('--output', dest='o', help='Enter the output folder path')
    args = parser.parse_args()
    if not os.path.isdir(args.o):
        os.makedirs(args.o)
    for root, dirs, files in os.walk(args.i):
        for fl in files:
            if fl.endswith('.tsv'):
                file_path = os.path.join(root, fl)
                input_data_frame = pd.read_csv(file_path, sep='\t', header='infer')
                file_name = fl[: fl.find('.')]
                tsv_file_path = os.path.join(args.o, file_name + '_with_filtered_comet_scores.tsv')
                filtered_data_frame = input_data_frame[input_data_frame["Comet_Scores"] < 0.6]
                filtered_data_frame.to_csv(tsv_file_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
