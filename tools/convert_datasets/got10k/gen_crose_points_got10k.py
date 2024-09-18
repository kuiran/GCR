import os
import random
import math

txt1 = 'annotations/got10k_test_infos.txt'
f = open(txt1, "r")  
lines = f.readlines()
for line in lines:  
    txt2 = line.split(',')[1]  
    if txt2.endswith("txt"):  
        root_dir = 'data/got10k/'  
        txt3 = root_dir + txt2.split('/', (1))[0] + '_rp/' + txt2.split('/', (1))[1]  
        dir1 = txt3.rsplit('/', (1))[0]  
        if not os.path.exists(dir1):  
            os.mkdir(dir1)

        gt = open(root_dir + txt2, "r")  
        gt_lines = gt.readlines()
        for gt_line in gt_lines:
            x_min = float(gt_line.split(',')[0]) 
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


            ctr = []
            ctr.append(str(random_point_x))
            ctr.append(str(random_point_y))
            # ctr.append(str(r))
            file_write_obj = open(txt3, 'a')  
            file_write_obj.write(','.join(ctr))  
            file_write_obj.write('\n')
            file_write_obj.close()
