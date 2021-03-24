import numpy as np
import os
import matplotlib.pyplot as plt

def gaussian2d_rolled_labels(sz,sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels = np.roll(labels, -int(np.floor(sz[1]/2)), axis=0)

    return labels

def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])

    return cos_window

def get_ground_truthes(img_path):
    gt_path = os.path.join(img_path, 'groundtruth_rect.txt')

    gts = []
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                gts = np.array(gts, dtype=np.float32)
                return gts
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos = line.split()
            gt_pos_int = [(float(element)) for element in gt_pos]
            gts.append(gt_pos_int)

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

def plot_all_success_for_tracking_methods(name, threshes_all, success_all, save_path=None):
    label = '[' + str(calAUC(success_all))[:5] + '] ' + name
    plt.plot(threshes_all, success_all, label=label)

    plt.title('Success Rate')
    plt.xlabel("overlap threshold")
    plt.ylabel("success rate")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_all_precision_for_tracking_methods(name, threshes_all, precision_all, save_path=None):
    idx20 = [i for i, x in enumerate(threshes_all) if x == 20][0]
    label = '[' + str(precision_all[idx20])[:5] + '] ' + name
    plt.plot(threshes_all, precision_all, label=label)

    plt.title('Distance Precision Rate')
    plt.xlabel("location error threshold (pixels)")
    plt.ylabel("precision rate")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_thresh_precision_pair(gts,preds):
    length = min(len(gts), len(preds))
    gts = gts[:length, :]
    preds = preds[:length, :]
    gt_centers_x = (gts[:, 0]+gts[:, 2]/2)
    gt_centers_y = (gts[:, 1]+gts[:, 3]/2)
    preds_centers_x = (preds[:, 0]+preds[:, 2]/2)
    preds_centers_y = (preds[:, 1]+preds[:, 3]/2)
    dists = np.sqrt((gt_centers_x - preds_centers_x) ** 2 + (gt_centers_y - preds_centers_y) ** 2)
    threshes = []
    precisions = []
    for thresh in np.linspace(0, 50, 101):
        true_len = len(np.where(dists < thresh)[0])
        precision = true_len / len(dists)
        threshes.append(thresh)
        precisions.append(precision)
    return threshes, precisions

def get_thresh_success_pair(gts, preds):
    length = min(len(gts), len(preds))
    gts = gts[:length, :]
    preds = preds[:length, :]
    intersect_tl_x = np.max((gts[:, 0], preds[:, 0]), axis=0)
    intersect_tl_y = np.max((gts[:, 1], preds[:, 1]), axis=0)
    intersect_br_x = np.min((gts[:, 0] + gts[:, 2], preds[:, 0] + preds[:, 2]), axis=0)
    intersect_br_y = np.min((gts[:, 1] + gts[:, 3], preds[:, 1] + preds[:, 3]), axis=0)
    intersect_w = intersect_br_x - intersect_tl_x
    intersect_w[intersect_w < 0] = 0
    intersect_h = intersect_br_y - intersect_tl_y
    intersect_h[intersect_h < 0] = 0
    intersect_areas = intersect_h * intersect_w
    ious = intersect_areas / (gts[:, 2] * gts[:, 3] + preds[:, 2] * preds[:, 3] - intersect_areas)
    threshes = []
    successes = []
    for thresh in np.linspace(0, 1, 101):
        success_len = len(np.where(ious > thresh)[0])
        success = success_len / len(ious)
        threshes.append(thresh)
        successes.append(success)
    return threshes, successes

def calAUC(value_list):
    length = len(value_list)
    delta = 1./(length-1)
    area = 0.
    for i in range(1, length):
        area += (delta*((value_list[i]+value_list[i-1])/2))
    return area

