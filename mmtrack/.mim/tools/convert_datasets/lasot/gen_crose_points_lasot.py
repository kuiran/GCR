import os
import random
import math

txt1 = '/mnt/dataset/kuiran/lasot/annotations/lasot_train_infos.txt'
f = open(txt1, 'r')
lines = f.readlines()
root_dir = '/mnt/dataset/kuiran/lasot/'
root_dir1 = '/mnt/dataset/kuiran/lasot/LaSOTBenchmark/'
lines.pop(0)
for line in lines:
    txt2 = line.split(',')[1]
    if txt2.endswith('txt'):
        txt3 = root_dir + 'lasot_train_points/' + txt2
        dir1 = txt3.rsplit('/', (1))[0]
        if not os.path.exists(dir1):
            os.makedirs(dir1)

        gt = open(root_dir1 + txt2, 'r')
        gt_lines = gt.readlines()
        for gt_line in gt_lines:
            x_min = float(gt_line.split(',')[0])  # 读取原数据
            y_min = float(gt_line.split(',')[1])
            w = float(gt_line.split(',')[2])
            h = float(gt_line.split(',')[3])

            center_x = x_min + w / 2
            center_y = y_min + h / 2
            a = w / 5
            b = h / 5

            if a == 0:
                random_point_x = 0.
                random_point_y = 0.
            else:
                _random_x = random.uniform(0, a)
                t_b_x = random.randint(0, 1)
                t_b_y = random.randint(0, 1)
                _random_y = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)

                if t_b_y == 0:
                    random_y = -_random_y
                elif t_b_y == 1:
                    random_y = _random_y

                if t_b_x == 0:
                    random_x = -_random_x
                elif t_b_x == 1:
                    random_x = _random_x

                random_point_x = round((center_x + random_x), 2)
                random_point_y = round((center_y + random_y), 2)

            # x_ctr = float(x_min) + (float(w) - 1) / 2  # 处理后新数据
            # y_ctr = float(y_min) + (float(h) - 1) / 2
            # r = 0.5 * min(float(w), float(h))

            ctr = []
            ctr.append(str(random_point_x))
            ctr.append(str(random_point_y))
            # ctr.append(str(r))
            file_write_obj = open(txt3, 'a')  # 打开新gt文件
            file_write_obj.write(','.join(ctr))  # 逐行写入新数据
            file_write_obj.write('\n')
            file_write_obj.close()