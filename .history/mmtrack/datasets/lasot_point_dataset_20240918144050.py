# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import os.path as osp
import time
import random

from mmdet.datasets import DATASETS
from mmcv.utils import print_log
from mmtrack.core.evaluation import eval_sot_ope

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class LaSOTPointDataset(BaseSOTDataset):
    """LaSOT dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self,
                 point_prefix_test,
                 point_prefix_train,
                 only_load_first_frame,
                 noisy_bbox_prefix,
                 with_track_bbox,
                 *args,
                 **kwargs):
        """Initialization of SOT dataset class."""
        super(LaSOTPointDataset, self).__init__(*args, **kwargs)
        # self.point_prefix = '/home/ubuntu/disk1/kuiran/dataset/lasot/lasot_test_ctr_1'
        # 140
        # self.point_prefix_test = '/mnt/dataset/kuiran/lasot/lasot_test_ctr_1'
        # self.point_prefix_train = '/mnt/dataset/kuiran/lasot/lasot_train_points'
        self.point_prefix_test = point_prefix_test
        self.point_prefix_train = point_prefix_train
        self.only_load_first_frame = only_load_first_frame
        self.noisy_bbox_prefix = noisy_bbox_prefix
        self.with_track_bbox = with_track_bbox
        if self.only_load_first_frame:
            self.num_frames_per_video = [1 for _ in range(len(self.data_infos))]

    def __getitem__(self, ind):
        if self.test_mode:
            assert isinstance(ind, tuple)
            # the first element in the tuple is the video index and the second
            # element in the tuple is the frame index
            if not self.with_track_bbox:
                return self.prepare_test_data(ind[0], ind[1])
            else:
                return self.prepare_test_data_with_track_bbox(ind[0], ind[1])
        else:
            return self.prepare_train_data(ind)

    def get_point_from_video(self, video_ind, mode='train'):
        if mode == 'test':
            point_path = osp.join(self.point_prefix_test, self.data_infos[video_ind]['ann_path'])
        elif mode == 'train':
            point_path = osp.join(self.point_prefix_train, self.data_infos[video_ind]['ann_path'])
        else:
            raise NotImplementedError
        point = self.loadtxt(point_path, dtype=float, delimiter=',').reshape(-1, 2)
        return point

    def get_noisy_bbox_from_video(self, video_ind):
        noisy_path = os.path.join(self.noisy_bbox_prefix, self.data_infos[video_ind]['ann_path'])
        noisy_bbox = self.loadtxt(noisy_path, dtype=float, delimiter=',').reshape(-1, 4)
        return noisy_bbox

    def get_ann_infos_from_video(self, video_ind, mode="train"):
        """Get annotation information in a video.

        Args:
            video_ind (int): video index

        Returns:
            dict: {'bboxes': ndarray in (N, 4) shape, 'bboxes_isvalid':
                ndarray, 'visible':ndarray}. The annotation information in some
                datasets may contain 'visible_ratio'. The bbox is in
                (x1, y1, x2, y2) format.
        """
        # class_info = self.data_infos[video_ind]['video_path'].split('/')[0]
        # del_classes = ['book', 'guitar', 'hand', 'helmet', 'licenseplate', 'microphone', 'swing', 'train']
        # if class_info in del_classes:
        #     return None
        # else:
        bboxes = self.get_bboxes_from_video(video_ind)
        point = self.get_point_from_video(video_ind, mode)

        # The visible information in some datasets may contain
        # 'visible_ratio'.
        visible_info = self.get_visibility_from_video(video_ind)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        visible_info['visible'] = visible_info['visible'] & bboxes_isvalid
        bboxes[:, 2:] += bboxes[:, :2]

        # noisy bbox
        if mode == 'test':
            noisy_bbox = self.get_noisy_bbox_from_video(video_ind)
            noisy_bbox[:, 2:] += noisy_bbox[:, :2]

            ann_infos = dict(
                bboxes=bboxes, noisy_bbox=noisy_bbox, points=point, bboxes_isvalid=bboxes_isvalid, **visible_info)
        else:
            ann_infos = dict(
                bboxes=bboxes, points=point, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    def get_ann_infos_from_video_refine_track_bbox(self, video_ind, mode="train"):
        """Get annotation information in a video.

        Args:
            video_ind (int): video index

        Returns:
            dict: {'bboxes': ndarray in (N, 4) shape, 'bboxes_isvalid':
                ndarray, 'visible':ndarray}. The annotation information in some
                datasets may contain 'visible_ratio'. The bbox is in
                (x1, y1, x2, y2) format.
        """
        # class_info = self.data_infos[video_ind]['video_path'].split('/')[0]
        # del_classes = ['book', 'guitar', 'hand', 'helmet', 'licenseplate', 'microphone', 'swing', 'train']
        # if class_info in del_classes:
        #     return None
        # else:
        bboxes = self.get_bboxes_from_video(video_ind)
        point = self.get_point_from_video(video_ind, mode)

        # The visible information in some datasets may contain
        # 'visible_ratio'.
        visible_info = self.get_visibility_from_video(video_ind)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        visible_info['visible'] = visible_info['visible'] & bboxes_isvalid
        # bboxes[:, 2:] += bboxes[:, :2]

        # noisy bbox
        if mode == 'test':
            noisy_bbox = self.get_noisy_bbox_from_video(video_ind)
            noisy_bbox[:, 2:] += noisy_bbox[:, :2]

            ann_infos = dict(
                bboxes=bboxes, noisy_bbox=noisy_bbox, points=point, bboxes_isvalid=bboxes_isvalid, **visible_info)
        else:
            ann_infos = dict(
                bboxes=bboxes, points=point, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    def load_data_infos(self, split='test'):
        """Load dataset information.

        Args:
            split (str, optional): Dataset split. Defaults to 'test'.

        Returns:
            list[dict]: The length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame number contained
                            in the image name
                        'end_frame_id': the ending frame number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading LaSOT dataset...')
        start_time = time.time()
        assert split in ['train', 'test']
        data_infos = []
        data_infos_str = self.loadtxt(
            self.ann_file, return_array=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            data_info = dict(
                video_path=line[0],
                ann_path=line[1],
                start_frame_id=int(line[2]),
                end_frame_id=int(line[3]),
                framename_template='%08d.jpg')
            data_infos.append(data_info)
        print(f'LaSOT dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_visibility_from_video(self, video_ind):
        """Get the visible information of instance in a video."""
        video_path = osp.dirname(self.data_infos[video_ind]['video_path'])
        full_occlusion_file = osp.join(self.img_prefix, video_path,
                                       'full_occlusion.txt')
        out_of_view_file = osp.join(self.img_prefix, video_path,
                                    'out_of_view.txt')
        full_occlusion = self.loadtxt(
            full_occlusion_file, dtype=bool, delimiter=',')
        out_of_view = self.loadtxt(out_of_view_file, dtype=bool, delimiter=',')
        visible = ~(full_occlusion | out_of_view)
        return dict(visible=visible)

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
            self.test_memo.ann_infos = self.get_ann_infos_from_video(video_ind, 'test')
            self.test_memo.img_infos = self.get_img_infos_from_video(video_ind)
        assert 'video_ind' in self.test_memo and 'ann_infos' in \
            self.test_memo and 'img_infos' in self.test_memo

        img_info = dict(
            filename=self.test_memo.img_infos['filename'][frame_ind],
            frame_id=frame_ind)
        ann_info = dict(
            bboxes=self.test_memo.ann_infos['bboxes'][frame_ind],
            noisy_bbox=self.test_memo.ann_infos['noisy_bbox'][0],
            points=self.test_memo.ann_infos['points'].squeeze().astype(np.float32),
            visible=self.test_memo.ann_infos['visible'][frame_ind])

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_test_data_with_track_bbox(self, video_ind, frame_ind):
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_ind (int): video index
            frame_ind (int): frame index

        Returns:
            dict: testing data of one frame.
        """
        # if self.test_memo.get('video_ind', None) != video_ind:
        #     self.test_memo.video_ind = video_ind
        #     self.test_memo.ann_infos = self.get_ann_infos_from_video(video_ind, 'test')
        #     self.test_memo.img_infos = self.get_img_infos_from_video(video_ind)
        # assert 'video_ind' in self.test_memo and 'ann_infos' in \
        #     self.test_memo and 'img_infos' in self.test_memo

        self.ann_infos = self.get_ann_infos_from_video(video_ind, 'test')
        self.img_infos = self.get_img_infos_from_video(video_ind)

        img_info = dict(
            filename=self.img_infos['filename'][frame_ind],
            frame_id=frame_ind)
        ann_info = dict(
            bboxes=self.ann_infos['bboxes'][frame_ind],
            noisy_bbox=self.ann_infos['noisy_bbox'][frame_ind],
            points=self.ann_infos['points'].squeeze().astype(np.float32),
            visible=self.ann_infos['visible'][frame_ind])

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

    # def prepare_train_data(self, video_ind):
    #     """Get training data sampled from some videos. We firstly sample two
    #     videos from the dataset and then parse the data information. The first
    #     operation in the training pipeline is frames sampling.
    #
    #     Args:
    #         video_ind (int): video index
    #
    #     Returns:
    #         dict: training data pairs, triplets or groups.
    #     """
    #     while True:
    #         # video_inds = random.choices(list(range(len(self))), k=2)
    #         pair_video_infos = []
    #         # for video_index in video_inds:
    #         ann_infos = self.get_ann_infos_from_video(video_ind, 'train')
    #         img_infos = self.get_img_infos_from_video(video_ind)
    #         video_infos = dict(**ann_infos, **img_infos)
    #         self.pre_pipeline(video_infos)
    #         pair_video_infos.append(video_infos)
    #
    #         results = self.pipeline(pair_video_infos)
    #         if results is not None:
    #             return results

    def evaluate(self, results, metric=['track'], logger=None):
        """Default evaluation standard is OPE.

        Args:
            results (dict(list[ndarray])): tracking results. The ndarray is in
                (x1, y1, x2, y2, score) format.
            metric (list, optional): defaults to ['track'].
            logger (logging.Logger | str | None, optional): defaults to None.
        """

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        # get all test annotations
        gt_bboxes = []
        visible_infos = []
        for video_ind in range(len(self.data_infos)):
            video_anns = self.get_ann_infos_from_video(video_ind, 'test')
            gt_bboxes.append(video_anns['bboxes'])
            visible_infos.append(video_anns['visible'])

        # tracking_bboxes converting code
        eval_results = dict()
        if 'track' in metrics:
            assert len(self) == len(
                results['track_bboxes']
            ), f"{len(self)} == {len(results['track_bboxes'])}"
            print_log('Evaluate OPE Benchmark...', logger=logger)
            track_bboxes = []
            start_ind = end_ind = 0
            for num in self.num_frames_per_video:
                end_ind += num
                track_bboxes.append(
                    list(
                        map(lambda x: x[:-1],
                            results['track_bboxes'][start_ind:end_ind])))
                start_ind += num

            if not self.only_eval_visible:
                visible_infos = None
            # evaluation
            track_eval_results = eval_sot_ope(
                results=track_bboxes,
                annotations=gt_bboxes,
                visible_infos=visible_infos)
            eval_results.update(track_eval_results)

            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{(v):.3f}')
            print_log(eval_results, logger=logger)
        return eval_results
