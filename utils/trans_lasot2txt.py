import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='lasot2txt')
    parser.add_argument('--lasot_ann_path')
    parser.add_argument('--lasot_ann_name')
    parser.add_argument('--save_path')
    parser.add_argument('--save_name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_txt = os.path.join(args.lasot_ann_path, 'annotations/lasot_test_infos.txt')
    with open(test_txt) as f:
        test_seq_name = f.read().split('\n')
    test_seq_name = test_seq_name[1:]
    noisy_bbox = []
    for _seq_name in test_seq_name:
        gt_path = _seq_name.split(',')[1]
        gt_path_ = os.path.join(args.lasot_ann_path, args.lasot_ann_name, gt_path)
        with open(gt_path_) as f:
            noisy_bbox_ = f.read()
        noisy_bbox.append(noisy_bbox_)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, args.save_name), 'w') as f:
        for nb in noisy_bbox:
            a = nb.split(',')
            for i in range(len(a)):
                a[i] = a[i].split('.')[0]
            a[2] = str(int(a[0]) + int(a[2]))
            a[3] = str(int(a[1]) + int(a[3]))
            new_nb = a[0] + ',' + a[1] + ',' + a[2] + ',' + a[3]
            f.write(new_nb + '\n')
    print('Done!')


if __name__ == '__main__':
    main()
