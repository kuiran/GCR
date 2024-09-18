# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time

from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class LaSOTDataset(BaseSOTDataset):
    """LaSOT dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, replace_first_frame_ann, first_frame_ann_path, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super(LaSOTDataset, self).__init__(*args, **kwargs)
        self.replace_first_frame_ann = replace_first_frame_ann
        self.first_frame_ann_path = first_frame_ann_path

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

    def get_replace_first_frame_bbox(self, video_ind):
        """
        Args:
            video_ind: int
        Return:
            bbox: ndarray: (1, 4)
        """
        bboxes = self.loadtxt(self.first_frame_ann_path, dtype=float, delimiter=',')
        # bboxes[:, 2:] += bboxes[:, :2]
        return bboxes[video_ind]

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
            self.test_memo.ann_infos = self.get_ann_infos_from_video(video_ind)
            if self.replace_first_frame_ann:
                self.test_memo.ann_infos['bboxes'][0] = self.get_replace_first_frame_bbox(video_ind)
            self.test_memo.img_infos = self.get_img_infos_from_video(video_ind)
        assert 'video_ind' in self.test_memo and 'ann_infos' in \
            self.test_memo and 'img_infos' in self.test_memo

        img_info = dict(
            filename=self.test_memo.img_infos['filename'][frame_ind],
            frame_id=frame_ind)
        ann_info = dict(
            bboxes=self.test_memo.ann_infos['bboxes'][frame_ind],
            visible=self.test_memo.ann_infos['visible'][frame_ind])

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results
