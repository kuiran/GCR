import cv2
import numpy as np
import copy
import os


class Visualize(object):

    def draw(self, img, graph, color):
        """
        img_meta: dict
        graph: ndarray(N, 4) or ndarray(N, 2) N:num for one image int
        """
        if graph.shape[1] == 4:
            for i in range(graph.shape[0]):
                img = cv2.rectangle(img, (graph[i, 0], graph[i, 1]), (graph[i, 2], graph[i, 3]), color=color)
        elif graph.shape[1] == 2:
            for i in range(graph.shape[0]):
                img = cv2.circle(img, (graph[i, 0], graph[i, 1]), 2, color, 4)
        else:
            raise NotImplementedError

        return img

    def __call__(self, img_metas, graphs, pre_fix, color=(0, 255, 255)):
        """
        img_metas: [img_meta]
        graphs: arrays (B, num, 4) or (B, num, 2)
        """
        for ind, img_meta in enumerate(img_metas):
            file_name = img_meta['filename']
            img_name = file_name.split('/')[-1]
            video_name = file_name.split('/')[-2]
            h, w, _ = img_meta['img_shape']
            img = cv2.imread(file_name)
            img = cv2.resize(img, (w, h))
            out_put_path = pre_fix + video_name
            if not os.path.exists(out_put_path):
                os.mkdir(out_put_path)

            img = self.draw(img, graphs[ind], color)
            cv2.imwrite(out_put_path + img_name, img)




# prefix = 'xywhep12614'
# gt_vis = torch.stack(gt_bboxes_, dim=0).squeeze(1)
# gt_vis_1 = np.array(gt_vis.cpu().squeeze(0)).astype(np.int32)
# pred_bbox_vis = self.vis_box.squeeze(0).clone().cpu().numpy().astype(np.int32)
# ppppp = random_points[0].squeeze(0).clone().cpu().numpy().astype(np.int32)
# file_name = img_metas[0]['filename']
# # frame_id = i_m['frame_id']
# video_name = file_name.split('/')[-2]
# h, w, _ = img_metas[0]['img_shape']
# img1 = cv2.imread(file_name)
# img1 = cv2.resize(img1, (w, h))
# igs1 = copy.deepcopy(img1)
# if not os.path.isdir('exp/{}'.format(prefix)):
#     os.mkdir('exp/{}'.format(prefix))
# igs1 = cv2.rectangle(igs1, (pred_bbox_vis[0], pred_bbox_vis[1]),
#                      (pred_bbox_vis[2], pred_bbox_vis[3]),
#                      color=(0, 256, 0))
# igs1 = cv2.circle(igs1, (ppppp[0], ppppp[1]), 2, (0, 0, 255), 4)
# igs1 = cv2.rectangle(igs1, (gt_vis_1[0], gt_vis_1[1]),
#                      (gt_vis_1[2], gt_vis_1[3]),
#                      color=(0, 0, 0))
# cv2.imwrite('exp/{}/{}_{}.jpg'.format(prefix, video_name, frame_id), igs1)