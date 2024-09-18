import os
import os.path as osp
import shutil
import time

import random
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset
from . import GOT10kDataset


@DATASETS.register_module()
class GOT10kPointDataset(GOT10kDataset):

    def __init__(self, only_load_first_frame, *args, **kwargs):
        super(GOT10kPointDataset, self).__init__(*args, **kwargs)
        self.only_load_first_frame = only_load_first_frame
        if self.only_load_first_frame:
            self.num_frames_per_video = [1 for _ in range(len(self.data_infos))]

    def get_bboxes_from_video(self, video_ind, mode):
        """Get bboxes annotation about the instance in a video.

        Args:
            video_ind (int): video index

        Returns:
            ndarray: in [N, 4] shape. The N is the number of bbox and the bbox
                is in (x, y, w, h) format.
        """
        bbox_path = osp.join(self.img_prefix,
                             self.data_infos[video_ind]['ann_path'])
        bbox_path_ = bbox_path.split('/')
        if mode == 'train':
            bbox_path_[-3] = '{}_rp'.format(mode)
        elif mode == 'test':
            bbox_path_[-3] = '{}_ctr'.format(mode)
        else:
            assert mode in ['train', 'test']
        point_path = '/'.join(bbox_path_)
        bboxes = self.loadtxt(bbox_path, dtype=float, delimiter=',')
        points = self.loadtxt(point_path, dtype=float, delimiter=',')[:2]
        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)

        end_frame_id = self.data_infos[video_ind]['end_frame_id']
        start_frame_id = self.data_infos[video_ind]['start_frame_id']

        if not self.test_mode:
            assert len(bboxes) == (
                end_frame_id - start_frame_id + 1
            ), f'{len(bboxes)} is not equal to {end_frame_id}-{start_frame_id}+1'  # noqa
            assert len(points) == (
                    end_frame_id - start_frame_id + 1
            ), f'{len(points)} is not equal to {end_frame_id}-{start_frame_id}+1'
        return bboxes, points

    def get_ann_infos_from_video(self, video_ind, mode):
        """Get annotation information in a video.

        Args:
            video_ind (int): video index

        Returns:
            dict: {'bboxes': ndarray in (N, 4) shape, 'bboxes_isvalid':
                ndarray, 'visible':ndarray}. The annotation information in some
                datasets may contain 'visible_ratio'. The bbox is in
                (x1, y1, x2, y2) format.
        """
        bboxes, points = self.get_bboxes_from_video(video_ind, mode)
        # The visible information in some datasets may contain
        # 'visible_ratio'.
        visible_info = self.get_visibility_from_video(video_ind)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
                bboxes[:, 3] > self.bbox_min_size)
        visible_info['visible'] = visible_info['visible'] & bboxes_isvalid
        bboxes[:, 2:] += bboxes[:, :2]
        ann_infos = dict(
            bboxes=bboxes, points=points, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    def prepare_test_data(self, video_ind, frame_ind):
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_ind (int): video index
            frame_ind (int): frame index

        Returns:
            dict: testing data of one frame.
        """
        if self.test_memo.get('video_ind', None) != video_ind:
            self.test_memo.video_ind = video_ind
            self.test_memo.img_infos = self.get_img_infos_from_video(video_ind)
        assert 'video_ind' in self.test_memo and 'img_infos' in self.test_memo

        img_info = dict(
            filename=self.test_memo.img_infos['filename'][frame_ind],
            frame_id=frame_ind)
        if frame_ind == 0:
            ann_infos = self.get_ann_infos_from_video(video_ind, 'test')
            ann_info = dict(
                bboxes=ann_infos['bboxes'][frame_ind], points=ann_infos['points'].astype(np.float32), visible=True)
        else:
            ann_info = dict(
                bboxes=np.array([0] * 4, dtype=np.float32), points=np.array([0] * 2, dtype=np.float32), visible=True)

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_train_data(self, video_ind):
        """Get training data sampled from some videos. We firstly sample two
        videos from the dataset and then parse the data information. The first
        operation in the training pipeline is frames sampling.

        Args:
            video_ind (int): video index

        Returns:
            dict: training data pairs, triplets or groups.
        """
        while True:
            video_inds = random.choices(list(range(len(self))), k=2)
            pair_video_infos = []
            for video_index in video_inds:
                ann_infos = self.get_ann_infos_from_video(video_index, 'train')
                img_infos = self.get_img_infos_from_video(video_index)
                video_infos = dict(**ann_infos, **img_infos)
                self.pre_pipeline(video_infos)
                pair_video_infos.append(video_infos)

            results = self.pipeline(pair_video_infos)
            if results is not None:
                return results