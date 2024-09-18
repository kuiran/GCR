# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from mmtrack.core import results2outs


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """Load multi images from file.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.

        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Sequence load annotations.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadAnnotations`
    for detailed docstring.

    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)

        # kuiran vis
        # import cv2
        # import numpy as np
        # import copy
        # import os
        # for i, i_m in enumerate(outs):
        #     file_name = i_m['filename']
        #     frame_id = i_m['img_info']['frame_id']
        #     video_name = file_name.split('/')[-2]
        #     img1 = cv2.imread(file_name)
        #     box_vis = i_m['gt_bboxes'].astype(np.int32)
        #     # h, w, _ = i_m['img_shape']
        #     # img1 = cv2.resize(img1, (w, h))
        #     igs1 = copy.deepcopy(img1)
        #     if not os.path.isdir('exp/pipeline/{}'.format(video_name)):
        #         os.mkdir('exp/pipeline/{}'.format(video_name))
        #     igs1 = cv2.rectangle(igs1, (box_vis[0, 0], box_vis[0, 1]), (box_vis[0, 2], box_vis[0, 3]), \
        #                          color=(0, 256, 0))
        #     cv2.imwrite('exp/pipeline/{}/{}.jpg'.format(video_name, frame_id), igs1)

        # kuiran
        for i in range(len(outs)):
            outs[i]['bbox_fields'] = outs[i]['bbox_fields'][:1]
        return outs


@PIPELINES.register_module()
class LoadDetections(object):
    """Load public detections from MOT benchmark.

    Args:
        results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.
    """

    def __call__(self, results):
        outs_det = results2outs(bbox_results=results['detections'])
        bboxes = outs_det['bboxes']
        labels = outs_det['labels']

        results['public_bboxes'] = bboxes[:, :4]
        if bboxes.shape[1] > 4:
            results['public_scores'] = bboxes[:, -1]
        results['public_labels'] = labels
        results['bbox_fields'].append('public_bboxes')
        return results
