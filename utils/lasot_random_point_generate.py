import os
import random
import math
from mmdet.core import bbox_xyxy_to_cxcywh


# def gen_point(bbox):
#     """
#     Args:
#         bboxes: list[tensor(1, 4)]
#     Return:
#         random_points: tensor(b, 2)
#     """
#     random_points = []
#     cx = bbox[0] + bbox[2] / 2
#     cy = bbox[1] + bbox[3] / 2
#     w = bbox[2]
#     h = bbox[3]

#     # cxcywh = bbox_xyxy_to_cxcywh(bbox)
#     # cx = cxcywh[:, 0].item()
#     # cy = cxcywh[:, 1].item()
#     # w = cxcywh[:, 2].item()
#     # h = cxcywh[:, 3].item()

#     a = w / 4
#     b = h / 4

#     if a == 0:
#         random_point_x = 0.
#         random_point_y = 0.
#     else:
#         _random_x = random.uniform(0, a)
#         t_b_x = random.randint(0, 1)
#         t_b_y = random.randint(0, 1)
#         _random_y = math.sqrt(b ** 2 - ((b ** 2) * (_random_x ** 2)) / a ** 2)

#         if t_b_y == 0:
#             random_y = -_random_y
#         elif t_b_y == 1:
#             random_y = _random_y

#         if t_b_x == 0:
#             random_x = -_random_x
#         elif t_b_x == 1:
#             random_x = _random_x

#         random_point_x = round((cx + random_x), 2)
#         random_point_y = round((cy + random_y), 2)
#     random_points.append(random_point_x)
#     random_points.append(random_point_y)
#     return random_points

"""
new_generation
"""
def gen_point(bbox):
    """
    Args:
        bboxes: list[tensor(1, 4)]
    Return:
        random_points: tensor(b, 2)
    """
    random_points = []
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    w = bbox[2]
    h = bbox[3]

    a = w / 4
    b = h / 4

    if a == 0:
        random_point_x = 0.
        random_point_y = 0.
    else:
        _random_x = random.uniform(-a, a)
        _random_y_range = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)
        _random_y = random.uniform(-_random_y_range, _random_y_range)
    random_x = _random_x
    random_y = _random_y

    random_point_x = round((cx + random_x), 2)
    random_point_y = round((cy + random_y), 2)

    random_points.append(random_point_x)
    random_points.append(random_point_y)
    return random_points

    
    # for bbox in bboxes:
    #     cxcywh = bbox_xyxy_to_cxcywh(bbox)
    #     cx = cxcywh[:, 0].item()
    #     cy = cxcywh[:, 1].item()
    #     w = cxcywh[:, 2].item()
    #     h = cxcywh[:, 3].item()

    #     a = w / 4
    #     b = h / 4

    #     if a == 0:
    #         random_point_x = 0.
    #         random_point_y = 0.
    #     else:
    #         _random_x = random.uniform(-a, a)
    #         _random_y_range = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)
    #         _random_y = random.uniform(-_random_y_range, _random_y_range)

    #         # _random_x = random.uniform(0, a)
    #         # t_b_x = random.randint(0, 1)
    #         # t_b_y = random.randint(0, 1)
    #         # _random_y = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)
    #         #
    #         # if t_b_y == 0:
    #         #     random_y = -_random_y
    #         # elif t_b_y == 1:
    #         #     random_y = _random_y
    #         #
    #         # if t_b_x == 0:
    #         #     random_x = -_random_x
    #         # elif t_b_x == 1:
    #         #     random_x = _random_x
    #         random_x = _random_x
    #         random_y = _random_y

    #         random_point_x = round((cx + random_x), 2)
    #         random_point_y = round((cy + random_y), 2)
    #     random_points.append(torch.tensor([[random_point_x, random_point_y]]))
    # random_points = torch.cat(random_points).unsqueeze(1)
    # return random_points

def main():
    with open('data/lasot/annotations/lasot_test_infos.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = data[1:]
    f.close()
    for index, ann in enumerate(data):
        data[index] = data[index].strip('\n').split(',')[1]
    prefix = 'data/lasot/LaSOTBenchmark'
    output_prefix = 'data/lasot/random_point_new_generation/10'
    for ann in data:
        with open(os.path.join(prefix, ann)) as f:
            frame1_gt_box = f.readlines()[0].strip('\n')
        frame1_gt_box = frame1_gt_box.split(',')
        for ind, s in enumerate(frame1_gt_box):
            frame1_gt_box[ind] = int(s)
        f.close()
        point = gen_point(frame1_gt_box)
        # write point
        output_file = os.path.join(output_prefix, ann)
        output_dir = os.path.split(output_file)[0]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # if not os.path.isdir(output_dir):
        #     os.makedirs(output_dir)
        with open(output_file, 'a') as f:
            for indd, p in enumerate(point):
                if indd == 1:
                    f.write(str(p) + '\n')
                else:
                    f.write(str(p) + ',')


if __name__ == '__main__':
    main()