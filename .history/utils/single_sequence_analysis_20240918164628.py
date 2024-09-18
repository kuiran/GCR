import os
import argparse
import pickle
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='pkl2txt')
    parser.add_argument('--test_result_result_path')
    parser.add_argument('--precise_result_path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_result = open(args.test_result_result_path, 'rb')
    test_result1 = json.load(test_result)
    test_single_sequence_success = test_result1['single_sequence_success']
    test_result.close()

    precise_result = open(args.precise_result_path, 'rb')
    precise_result1 = json.load(precise_result)
    precise_single_sequence_success = precise_result1['single_sequence_success']
    precise_result.close()

    seq_name_path = 'data/lasot/annotations/lasot_test_infos.txt'
    with open(seq_name_path, 'r') as f:
        line = f.readlines()
    line = line[1:]
    seq_name_list = [_.split('/')[-2] for _ in line]

    precise_single_sequence_success_ = np.array(precise_single_sequence_success)
    test_single_sequence_success_ = np.array(test_single_sequence_success)

    single_sub = precise_single_sequence_success_ - test_single_sequence_success_
    overall_sub = (precise_single_sequence_success_ - test_single_sequence_success_) / 280
    for seq_name, precise_s, test_s, single_s, overall_s in zip(seq_name_list, precise_single_sequence_success,
                                                                test_single_sequence_success, single_sub.tolist(),
                                                                overall_sub.tolist()):
        print('{}:{},{},{},{}'.format(seq_name, precise_s, test_s, single_s, overall_s))


if __name__ == '__main__':
    main()