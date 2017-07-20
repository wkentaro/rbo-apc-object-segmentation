#!/usr/bin/env python

import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import tqdm

from rbo_seg.apc_data import APCDataSet
from rbo_seg.apc_data import APCSample
from rbo_seg.seg import ProbabilisticSegmentationBP


def evaluate(bp, test_data, with_candidates=False):
    imgs = []
    lbl_preds = []
    lbl_trues = []
    for sample in tqdm.tqdm(test_data.samples):
        if not with_candidates:
            sample.candidate_objects = test_data.object_names

        bp.predict(sample, desired_object='shelf')  # no candidates

        proba_img = []
        for obj in test_data.object_names:
            proba_img.append(bp.posterior_images_smooth[obj])
        lbl_pred = np.argmax(proba_img, axis=0).astype(np.int32)

        lbl_true = np.zeros_like(lbl_pred)
        for obj_id, obj in enumerate(test_data.object_names):
            if obj != 'shelf' and obj in sample.object_masks:
                lbl_true[sample.object_masks[obj]] = obj_id

        lbl_pred[~sample.bin_mask] = -1
        lbl_true[~sample.bin_mask] = -1

        img = skimage.io.imread(sample.file_prefix + '.jpg')
        x1, y1 = sample.bounding_box['x'], sample.bounding_box['y']
        x2 = x1 + sample.bounding_box['w']
        y2 = y1 + sample.bounding_box['h']
        img = img[y1:y2, x1:x2]

        imgs.append(img)
        lbl_preds.append(lbl_pred)
        lbl_trues.append(lbl_true)

    n_class = len(test_data.object_names)
    acc, acc_cls, mean_iu, fwavacc = fcn.utils.label_accuracy_score(
        lbl_trues, lbl_preds, n_class)
    print('Acc: %.2f' % (acc * 100))
    print('AccCls: %.2f' % (acc_cls * 100))
    print('MeanIU: %.2f' % (mean_iu * 100))
    print('FWAVACC: %.2f' % (fwavacc * 100))

    return imgs, lbl_trues, lbl_preds


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


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out', default=osp.join(here, 'logs/visualizations_evaluate'))
    args = parser.parse_args()

    out = args.out

    dataset_path = osp.expanduser('~/data/datasets/APC2016/APC2016rbo')
    dataset = create_dataset(dataset_path)
    dataset_train, dataset_test = dataset.split_simple(portion_training=0.7)

    params = {
        'use_features': ['color', 'edge'],
        'segmentation_method': "max_smooth",
        'selection_method': "max_smooth",
        'make_convex': True,
        'do_shrinking_resegmentation': True,
        'do_greedy_resegmentation': True,
    }
    clf = ProbabilisticSegmentationBP(**params)
    print('Fitting the segmenter.')
    clf.fit(dataset_train)

    print('Evaluating with test set.')
    imgs, lbl_trues, lbl_preds = evaluate(
        clf, dataset_test, with_candidates=False)

    print('Saving visualizations: %s' % out)
    if not osp.exists(out):
        os.makedirs(out)
    for i, (img, lbl_true, lbl_pred) in \
            enumerate(zip(imgs, lbl_trues, lbl_preds)):
        viz = fcn.utils.visualize_segmentation(
            lbl_true=lbl_true, lbl_pred=lbl_pred, img=img,
            n_class=len(dataset_test.object_names))
        out_file = osp.join(out, '%06d.jpg' % i)
        skimage.io.imsave(out_file, viz)


if __name__ == '__main__':
    main()
