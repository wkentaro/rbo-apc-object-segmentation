from apc_data import APCDataSet, APCSample
from probabilistic_segmentation import ProbabilisticSegmentationRF, ProbabilisticSegmentationBP
import pickle
import os

import fcn
import matplotlib.pyplot as plt
import numpy as np
import copy


def evaluate(bp, test_data):
    lbl_preds = []
    lbl_trues = []
    for sample in test_data.samples:
        sample.candidate_objects = test_data.object_names

        if len(sample.object_masks) == 0:
            continue
        pred_target = sample.object_masks.keys()[0]
        if pred_target == 'shelf':
            if len(sample.object_masks.keys()) == 1:
                continue
            pred_target = sample.object_masks.keys()[1]
        bp.predict(sample, pred_target)

        images = []
        for _object in test_data.object_names:
            if _object in bp.posterior_images_smooth:
                images.append(bp.posterior_images_smooth[_object])
            else:
                raise ValueError
                images.append(np.zeros_like(images[0]))
        pred = np.argmax(np.array(images), axis=0).astype(np.int32)

        true = np.zeros_like(pred)
        for obj_id, obj in enumerate(test_data.object_names):
            if obj != 'shelf' and obj in sample.object_masks:
                true[sample.object_masks[obj]] = obj_id

        pred[~sample.bin_mask] = -1
        true[~sample.bin_mask] = -1

        lbl_preds.append(pred)
        lbl_trues.append(true)

        # import mvtk
        # import cv2
        # image = cv2.imread(sample.file_prefix + '.jpg')[:, :, ::-1]
        # x1 = sample.bounding_box['x']
        # y1 = sample.bounding_box['y']
        # x2 = x1 + sample.bounding_box['w']
        # y2 = y1 + sample.bounding_box['h']
        # image = image[y1:y2, x1:x2]
        # image[~sample.bin_mask] = 0
        # #
        # pred_viz = mvtk.image.label2rgb(pred)
        # true_viz = mvtk.image.label2rgb(true)
        # #
        # pred_viz = mvtk.image.tile([image, pred_viz], shape=(1, 2))
        # true_viz = mvtk.image.tile([image, true_viz], shape=(1, 2))
        # viz = mvtk.image.tile([true_viz, pred_viz], shape=(2, 1))
        # #
        # cv2.imshow('viz', viz[:, :, ::-1])
        # cv2.waitKey(0)
    n_class = len(test_data.object_names)
    return fcn.utils.label_accuracy_score(lbl_trues, lbl_preds, n_class)


def create_dataset(dataset_path):
    # initialize empty dataset
    dataset = APCDataSet(from_pkl=False)

    data_file_prefixes = []
    key = '.jpg'
    for dir_name, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if key == f[-len(key):]:
                data_file_prefixes.append(
                    os.path.join(dir_name, f[:-len(key)]))

    for file_prefix in data_file_prefixes:
        dataset.samples.append(
            APCSample(data_2016_prefix=file_prefix,
                        labeled=True, is_2016=True, infer_shelf_mask=True))
    return dataset


def main():
    dataset_path = '/home/wkentaro/data/datasets/APC2016/APC2016rbo'
    dataset = create_dataset(dataset_path)
    dataset_train, dataset_valid = dataset.split_simple(portion_training=0.7)

    params = {
        'use_features': ['color'],
        'segmentation_method': "max_smooth",
        'selection_method': "max_smooth",
        'make_convex': True,
        'do_shrinking_resegmentation': True,
        'do_greedy_resegmentation': True,
    }
    clf = ProbabilisticSegmentationBP(**params)
    clf.fit(dataset_train)
    acc, acc_cls, mean_iu, fwavacc = evaluate(clf, dataset_valid)

    print 'trained only by color features acc ', acc
    print 'trained only by color features acc_cls ', acc_cls
    print 'trained only by color features mean_iu ', mean_iu
    print 'trained only by color features fwavcc ', fwavacc


if __name__ == '__main__':
    main()
