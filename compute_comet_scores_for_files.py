"""Compute COMET scores on parallel sentences from files."""
from argparse import ArgumentParser
import os
from comet import download_model, load_from_checkpoint
import pandas as pd
from collections import OrderedDict


# Choose a HuggingFace model without reference
model_path_without_reference = download_model("Unbabel/wmt22-cometkiwi-da")
# Load the model checkpoint:
model_no_ref = load_from_checkpoint(model_path_without_reference)


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def create_data_in_comet_format(lines):
    """Create data ib COMET format from lines read from files."""
    data = []
    for line in lines:
        if len(line.split('\t')) >= 2:
            src, tgt = line.split('\t')[: 2]
            dict_line = {"src": src, "mt": tgt}
            data.append(dict_line)
    return data


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
            file_path = os.path.join(root, fl)
            lines = read_lines_from_file(file_path)
            comet_formatted_data = create_data_in_comet_format(lines)
            data_with_scores = []
            comet_scores = model_no_ref.predict(comet_formatted_data, batch_size=8, gpus=1)
            sentence_wise_comet_scores = comet_scores.scores
            score_dist = OrderedDict({'< 0.50': 0, '>= 0.50 and < 0.60': 0, '>= 0.60 and < 0.70': 0, '>= 0.70 and < 0.80': 0, '>= 0.80 and < 0.90': 0, '>= 0.90': 0})
            for i in range(len(comet_formatted_data)):
                dict_obj = comet_formatted_data[i]
                comet_score = sentence_wise_comet_scores[i]
                if comet_score < 0.5:
                    score_dist['< 0.50'] += 1
                elif comet_score >= 0.5 and comet_score < 0.6:
                    score_dist['>= 0.50 and < 0.60'] += 1
                elif comet_score >= 0.6 and comet_score < 0.7:
                    score_dist['>= 0.60 and < 0.70'] += 1
                elif comet_score >= 0.7 and comet_score < 0.8:
                    score_dist['>= 0.70 and < 0.80'] += 1
                elif comet_score >= 0.8 and comet_score < 0.9:
                    score_dist['>= 0.80 and < 0.90'] += 1
                elif comet_score >= 0.9:
                    score_dist['>= 0.90'] += 1
                dict_obj["comet_score"] = comet_score
                info_list = [dict_obj["src"], dict_obj["mt"], dict_obj["comet_score"]]
                data_with_scores.append(info_list)
            file_name = fl[: fl.find('.')]
            tsv_file_path = os.path.join(args.o, file_name + '_with_comet_scores.tsv')
            dist_file_path = os.path.join(args.o, file_name + '_with_comet_scores_distribution.txt')
            list_of_score_dist = [key + '\t' + str(val) for key, val in score_dist.items()]
            list_of_score_dist += ['Total\t' + str(sum(score_dist.values()))]
            write_lines_to_file(list_of_score_dist, dist_file_path)
            output_frame = pd.DataFrame(data_with_scores, columns=["Source", "Target", "Comet_Scores"])
            output_frame.to_csv(tsv_file_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
