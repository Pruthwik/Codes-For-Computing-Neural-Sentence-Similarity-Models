"""Compute LABSE score from a CSV file."""
from argparse import ArgumentParser
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
# Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
import numpy as np
import os
from collections import OrderedDict


preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


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
            score_dist = OrderedDict({'< 0.50': 0, '>= 0.50 and < 0.60': 0, '>= 0.60 and < 0.70': 0, '>= 0.70 and < 0.80': 0, '>= 0.80 and < 0.90': 0, '>= 0.90': 0})
            file_name = fl[: fl.rfind('.')]
            output_path = os.path.join(args.o, file_name + '.tsv')
            output_list = []
            for line in lines:
                source, target = line.split('\t')
                source_embeds = encoder(preprocessor(source))["default"]
                target_embeds = encoder(preprocessor(target))["default"]
                # For semantic similarity tasks, apply l2 normalization to embeddings
                source_embeds = normalization(source_embeds)
                target_embeds = normalization(target_embeds)
                labse_score = np.matmul(source_embeds, np.transpose(target_embeds))
                labse_score_only = labse_score[0][0]
                output_list.append([source, target, str(labse_score_only)])
                if labse_score_only < 0.5:
                    score_dist['< 0.50'] += 1
                elif labse_score_only >= 0.5 and labse_score_only < 0.6:
                    score_dist['>= 0.50 and < 0.60'] += 1
                elif labse_score_only >= 0.6 and labse_score_only < 0.7:
                    score_dist['>= 0.60 and < 0.70'] += 1
                elif labse_score_only >= 0.7 and labse_score_only < 0.8:
                    score_dist['>= 0.70 and < 0.80'] += 1
                elif labse_score_only >= 0.8 and labse_score_only < 0.9:
                    score_dist['>= 0.80 and < 0.90'] += 1
                elif labse_score_only >= 0.9:
                    score_dist['>= 0.90'] += 1
            output_frame = pd.DataFrame(output_list, columns=['Source', 'Target', 'LABSE_Score'])
            dist_file_path = os.path.join(args.o, file_name + '_with_bert_scores_distribution.txt')
            list_of_score_dist = [key + '\t' + str(val) for key, val in score_dist.items()]
            list_of_score_dist += ['Total\t' + str(sum(score_dist.values()))]
            write_lines_to_file(list_of_score_dist, dist_file_path)
            output_frame.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
