import os
import random
import math
# txt1 = '/home/ubuntu/mnt/dataset/kuiran/got10k/annotations/got10k_train_infos.txt'   #说明文件绝对路径
# txt1 = '/home/ubuntu/kuiran/dataset/got10k/annotations/got10k_train_infos.txt'
txt1 = '/home/ubuntu/kuiran/dataset/got10k/annotations/got10k_test_infos.txt'
f = open(txt1, "r")  # 读说明文件全部内容 ，并以列表方式返回
lines = f.readlines()
for line in lines:  # 读列表
    txt2 = line.split(',')[1]  # 原gt文件相对路径
    if txt2.endswith("txt"):  # 筛选txt文件
        root_dir = '/home/ubuntu/kuiran/dataset/got10k/'  # 根目录
        txt3 = root_dir + txt2.split('/', (1))[0] + '_rp/' + txt2.split('/', (1))[1]  # 预处理后新gt文件存放路径
        dir1 = txt3.rsplit('/', (1))[0]  # 新gt文件夹路径
        if not os.path.exists(dir1):  # 文件夹不存在则创建文件夹
            os.mkdir(dir1)

        gt = open(root_dir + txt2, "r")  # 读单个gt文件全部内容 ，并以列表方式返回
        gt_lines = gt.readlines()
        for gt_line in gt_lines:
            x_min = float(gt_line.split(',')[0])  # 读取原数据
            y_min = float(gt_line.split(',')[1])
            w = float(gt_line.split(',')[2])
            h = float(gt_line.split(',')[3])

            center_x = x_min + w / 2
            center_y = y_min + h / 2
            a = w / 4
            b = h / 4
            _random_x = random.uniform(0, a)
            t_b_x = random.randint(0, 1)
            t_b_y = random.randint(0, 1)
            if w > h:
                _random_y = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)
            else:
                _random_y = math.sqrt(a**2 - ((a**2) * (_random_x**2)) / b**2)

            if t_b_y == 0:
                random_y = -_random_y
            elif t_b_y == 1:
                random_y = _random_y

            if t_b_x == 0:
                random_x = -_random_x
            elif t_b_x == 1:
                random_x = _random_x

            random_point_x = round((center_x + random_x), 1)
            random_point_y = round((center_y + random_y), 1)

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
