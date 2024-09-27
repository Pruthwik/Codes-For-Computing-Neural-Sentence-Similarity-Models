"""Compute LABSE score from tab separated files containig source and target sentences."""
from argparse import ArgumentParser
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F
import os
from collections import OrderedDict


tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()


def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


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
            split_lines = list(map(lambda x: x.split('\t'), lines))
            source, target = list(zip(* split_lines))
            score_dist = OrderedDict({'< 0.50': 0, '>= 0.50 and < 0.60': 0, '>= 0.60 and < 0.70': 0, '>= 0.70 and < 0.80': 0, '>= 0.80 and < 0.90': 0, '>= 0.90': 0})
            file_name = fl[: fl.rfind('.')]
            output_path = os.path.join(args.o, file_name + '.tsv')
            output_list = []
            with torch.no_grad():
                source_inputs = tokenizer(source, return_tensors="pt", padding=True)
                source_outputs = model(**source_inputs)
                source_embeddings = source_outputs.pooler_output
                target_inputs = tokenizer(target, return_tensors="pt", padding=True)
                target_outputs = model(**target_inputs)
                target_embeddings = source_outputs.pooler_output
                labse_scores = torch.diagonal(similarity(source_embeddings, target_embeddings)).numpy().tolist()
                for src, tgt, labse_score in zip(source, target, labse_scores):
                    labse_score_str = str(labse_score)[: 5]
                    output_list.append([src, tgt, labse_score_str])
                    if labse_score < 0.5:
                        score_dist['< 0.50'] += 1
                    elif labse_score >= 0.5 and labse_score < 0.6:
                        score_dist['>= 0.50 and < 0.60'] += 1
                    elif labse_score >= 0.6 and labse_score < 0.7:
                        score_dist['>= 0.60 and < 0.70'] += 1
                    elif labse_score >= 0.7 and labse_score < 0.8:
                        score_dist['>= 0.70 and < 0.80'] += 1
                    elif labse_score >= 0.8 and labse_score < 0.9:
                        score_dist['>= 0.80 and < 0.90'] += 1
                    elif labse_score >= 0.9:
                        score_dist['>= 0.90'] += 1
                output_frame = pd.DataFrame(output_list, columns=['Source', 'Target', 'LABSE_Score'])
                dist_file_path = os.path.join(args.o, file_name + '_with_labse_scores_distribution.txt')
                list_of_score_dist = [key + '\t' + str(val) for key, val in score_dist.items()]
                list_of_score_dist += ['Total\t' + str(sum(score_dist.values()))]
                write_lines_to_file(list_of_score_dist, dist_file_path)
                output_frame.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
