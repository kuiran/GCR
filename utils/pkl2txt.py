import os
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='pkl2txt')
    parser.add_argument('--pkl_path')
    parser.add_argument('--txt_save_path')
    parser.add_argument('--txt_name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    f = open(args.pkl_path, 'rb')
    a = pickle.load(f)
    ori_pred_bboxes = a['ori_pred_bbox']
    ori_pred_bboxes_1 = [_.int().tolist() for _ in ori_pred_bboxes]
    txt_file = args.txt_save_path
    file_name = args.txt_name
    if not os.path.isdir(txt_file):
        os.makedirs(txt_file)
    with open(os.path.join(txt_file, file_name), 'w') as f:
        for bbox in ori_pred_bboxes_1:
            # bbox = bbox[0]
            for i in range(4):
                if i < 3:
                    f.write(str(bbox[0][i]) + ',')
                else:
                    f.write(str(bbox[0][i]) + '\n')
    f.close()


if __name__ == '__main__':
    main()