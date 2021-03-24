import os
from os.path import join, dirname, realpath
from kernelized_correlation_filter import HierarchicalKernelizedCorrelationFilter
from utils import get_ground_truthes, get_img_list, calAUC, plot_all_precision_for_tracking_methods, \
    plot_all_success_for_tracking_methods, get_thresh_precision_pair, get_thresh_success_pair
import numpy as np


def run():
    data_dir = 'dataset'
    data_dir = 'D:/DATASET/OTB100'
    data_names = sorted(os.listdir(data_dir))
    all_precisions_for_tracking_method = []
    all_thresh_p_for_tracking_method = []
    all_success_for_tracking_method = []
    all_thresh_s_for_tracking_method = []

    try:
        for data_name in data_names:
            tracker = HierarchicalKernelizedCorrelationFilter()

            data_path = join(data_dir, data_name)
            gts = get_ground_truthes(data_path)

            img_dir = os.path.join(data_path, 'img')
            frame_list = get_img_list(img_dir)
            frame_list.sort()
            preds = tracker.start(init_gt=gts[0], show=True, frame_list=frame_list)

            threshes_p, precisions = get_thresh_precision_pair(gts, preds)
            all_thresh_p_for_tracking_method.append(threshes_p)
            all_precisions_for_tracking_method.append(precisions)
            idx20 = [i for i, x in enumerate(threshes_p) if x == 20][0]
            print(data_name, '20px: ', str(round(precisions[idx20], 3)))

            threshes_s, successes = get_thresh_success_pair(gts, preds)
            all_thresh_s_for_tracking_method.append(threshes_s)
            all_success_for_tracking_method.append(successes)
            print(data_name, 'AUC', str(round(calAUC(successes), 3)))

        # compute average precision for tracking_method at 20px
        precision_mean = np.mean(all_precisions_for_tracking_method, axis=0)
        threshes_p_mean = np.mean(all_thresh_p_for_tracking_method, axis=0)
        idx20 = [i for i, x in enumerate(threshes_p_mean) if x == 20][0]
        print(' Average precision: ', str(round(precision_mean[idx20], 3)))

        # compute average success for tracking_method at 0.5
        success_mean = np.mean(all_success_for_tracking_method, axis=0)
        threshes_s_mean = np.mean(all_thresh_s_for_tracking_method, axis=0)
        print(' Average success AUC: ', str(round(calAUC(success_mean), 3)))

        plot_all_success_for_tracking_methods('HierarchicalKernelizedCorrelationFilter with VGG-19',
                                              threshes_s_mean, success_mean)
        plot_all_precision_for_tracking_methods('HierarchicalKernelizedCorrelationFilter with VGG-19',
                                                threshes_p_mean, precision_mean)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    run()
