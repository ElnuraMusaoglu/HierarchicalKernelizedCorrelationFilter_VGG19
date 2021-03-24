'''

Elnura Musaoglu
2021

'''

import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Keras version: {}".format(tf.keras.__version__))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=opts))

import cv2
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, fft2
from numpy import conj, real
from utils import gaussian2d_rolled_labels, cos_window
from scipy import io


ixs = [2, 5, 10, 15, 20]
indLayers = [2, 3, 4]
nweights = np.zeros([1, 1, 3])
nweights[0, 0, 0] = 0.5
nweights[0, 0, 1] = 1
nweights[0, 0, 2] = 0.25

vgg_path = 'D:/PROJECTS/MOT_CNN_DATAASSOCIATION/imagenet-vgg-verydeep-19.mat'  # model/imagenet-vgg-verydeep-19.mat

def create_model():

    from keras.applications.vgg19 import VGG19
    from keras.models import Model

    mat = io.loadmat(vgg_path)
    model = VGG19(mat)

    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    # model.summary()
    return model

vgg_model = create_model()

class HierarchicalKernelizedCorrelationFilter():
    def __init__(self):
        self.max_patch_size = 256
        self.padding = 1.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.output_sigma_factor = 0.1
        self.interp_factor = 0.01
        self.cell_size = 4
        self.resize = False
        self.model_xf = []
        self.model_x = []
        self.model_alphaf = []

    def start(self, init_gt, show, frame_list):
        poses = []
        poses.append(init_gt)
        init_frame = cv2.imread(frame_list[0])
        x1, y1, w, h = init_gt
        init_gt = tuple(init_gt)
        self.init(init_frame, init_gt)

        for idx in range(len(frame_list)):
            if idx != 0:
                current_frame = cv2.imread(frame_list[idx])
                bbox = self.update(current_frame)
                if bbox is not None:
                    x1, y1, w, h = bbox
                    if show is True:
                        if len(current_frame.shape) == 2:
                            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
                        show_frame = cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                                   (255, 0, 0), 1)
                        cv2.imshow('demo', show_frame)
                        cv2.waitKey(1)
                else:
                    print('bbox is None')
                poses.append(np.array([int(x1), int(y1), int(w), int(h)]))

        return np.array(poses)

    def init(self, image, roi):
        x, y, w, h = roi
        self.crop_size = (w, h)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.target_sz = np.array([h, w])
        self.target_sz_real = np.array([h, w])
        self.pos = np.array([y + np.floor(h / 2), x + np.floor(w / 2)])
        if np.sqrt(h * w) >= 100:  # diagonal size >= threshold
            self.resize = True
            self.pos = np.floor(self.pos / 2)
            self.target_sz = np.floor(self.target_sz / 2)
        if self.resize:
            self.image = cv2.resize(self.image, (self.image.shape[1] // 2, self.image.shape[0] // 2))
        self.window_sz = self.get_search_window()
        #  Compute the sigma for the Gaussian function label
        output_sigma = np.sqrt(self.target_sz[0] * self.target_sz[1]) * self.output_sigma_factor / self.cell_size
        # create regression labels, gaussian shaped, with a bandwidth
        self.l1_patch_num = np.floor(self.window_sz / self.cell_size)
        # Pre-compute the Fourier Transform of the Gaussian function label
        self.yf = fft2(gaussian2d_rolled_labels([self.l1_patch_num[1], self.l1_patch_num[0]], output_sigma))
        # Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
        self.cos_window = cos_window([self.yf.shape[1], self.yf.shape[0]])
        # Note: variables ending with 'f' are in the Fourier domain.
        self.model_xf = []
        self.model_x = []
        self.model_alphaf = []
        self.current_scale_factor = 1

        feat = self.extract_feature(self.image, self.pos, self.window_sz)
        self.update_model(feat, self.yf, self.interp_factor, self.lambdar, 0)

    def update(self, image, vis=False):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.resize:
            self.image = cv2.resize(self.image, (self.image.shape[1] // 2, self.image.shape[0] // 2))
        # Extracting hierarchical convolutional features
        feat = self.extract_feature(self.image, self.pos, self.window_sz)
        # Predict position
        pos = self.predict_position(feat, self.pos, indLayers, nweights, self.cell_size, self.l1_patch_num)
        # Extracting hierarchical convolutional features
        feat = self.extract_feature(self.image, pos, self.window_sz)
        self.update_model(feat, self.yf, self.interp_factor, self.lambdar, 1)
        target_sz_t = self.target_sz * self.current_scale_factor
        self.pos = pos

        if self.resize:
            pos_real = np.multiply(self.pos, 2)
        else:
            pos_real = self.pos

        box = [pos_real[1] - self.target_sz_real[1] / 2,
               pos_real[0] - self.target_sz_real[0] / 2,
               self.target_sz_real[1],
               self.target_sz_real[0]]

        return box[0], box[1], box[2], box[3]

    def extract_feature(self, im, pos, window_sz):
        # Get the search window from previous detection
        patch = self.get_subwindow(im, pos, window_sz)
        features = self.get_features(patch)

        return features

    def get_search_window(self):
        window_sz = np.floor(np.multiply(self.target_sz, (1 + self.padding)))

        return window_sz

    def get_subwindow(self, im, pos, sz):
        _p1 = np.array(range(0, int(sz[0]))).reshape([1, int(sz[0])])
        _p2 = np.array(range(0, int(sz[1]))).reshape([1, int(sz[1])])
        ys = np.floor(pos[0]) + _p1 - np.floor(sz[0] / 2)
        xs = np.floor(pos[1]) + _p2 - np.floor(sz[1] / 2)

        # Check for out-of-bounds coordinates, and set them to the values at the borders
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs > np.size(im, 1) - 1] = np.size(im, 1) - 1
        ys[ys > np.size(im, 0) - 1] = np.size(im, 0) - 1
        xs = xs.astype(int)
        ys = ys.astype(int)
        # extract image
        out1 = im[list(ys[0, :]), :, :]
        out = out1[:, list(xs[0, :]), :]

        return out

    def get_features(self, im):
        from numpy import expand_dims
        # img = im.astype('float32')  # note: [0, 255] range
        img = im  # note: [0, 255] range
        img = cv2.resize(img, (224, 224))
        average = np.zeros((224, 224, 3)).astype('float64')
        average[:, :, 0] = 123.6800  # RGB 123, BGR 103
        average[:, :, 1] = 116.7790
        average[:, :, 2] = 103.9390

        img = np.subtract(img, average)
        img = expand_dims(img, axis=0)
        # img = preprocess_input(img)

        feature_maps = vgg_model.predict(img)
        features = []

        for fi in range(len(indLayers)):
            f_map = feature_maps[indLayers[fi]][0][:][:][:]
            feature_map_n = cv2.resize(f_map, (self.cos_window.shape[1], self.cos_window.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

            feature_map_n = (feature_map_n - np.min(feature_map_n)) / (np.max(feature_map_n) - np.min(feature_map_n))
            feature_map_n = feature_map_n * self.cos_window[:, :, None]
            features.append(feature_map_n)

        return features

    def update_model(self, feat, yf, interp_factor, lambdar, frame):
        numLayers = len(feat)
        self.model_x = feat
        xf = []
        alphaf = []
        # Model update
        for i in range(numLayers):
            featf = fftn(feat[i], axes=(0, 1))
            xf.append(featf)
            kf = self.gaussian_correlation(featf, featf)
            alphaf.append(np.divide(yf, (kf + lambdar)))
        # Model initialization or update
        if frame == 0:
            self.model_alphaf = alphaf
            self.model_xf = xf
        else:
            for i in range(numLayers):
                self.model_alphaf[i] = (1 - interp_factor) * self.model_alphaf[i] + interp_factor * alphaf[i]
                self.model_xf[i] = (1 - interp_factor) * self.model_xf[i] + interp_factor * xf[i]

    def predict_position(self, feat, pos, indLayers, nweights, cell_size, l1_patch_num):
        # Compute correlation filter responses at each layer
        res_layer = np.zeros([int(l1_patch_num[0]), int(l1_patch_num[1]), len(indLayers)])

        for i in range(len(indLayers)):
            zf = fftn(feat[i], axes=(0, 1))
            kzf = self.gaussian_correlation(zf, self.model_xf[i])
            temp = real(ifftn(self.model_alphaf[i] * kzf, axes=(0, 1)))  # equation for fast detection
            temp = temp.real
            res_layer[:, :, i] = temp / (np.max(temp) + 0.00001)

        # Combine responses from multiple layers (see Eqn. 5)
        # response = sum(bsxfun(@times, res_layer, nweights), 3)
        response = np.sum(res_layer * nweights, axis=2)

        # Find target location
        # if the target doesn't move, the peak will appear at the top-left corner, not at the center.
        # The responses wrap around cyclically.

        # Find indices and values of nonzero elements
        delta = np.unravel_index(np.argmax(response, axis=None), response.shape)
        vert_delta, horiz_delta = delta[0], delta[1]
        if vert_delta > np.size(zf, 0) / 2:  # wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - np.size(zf, 0)
        if horiz_delta > np.size(zf, 1) / 2:  # same for horizontal axis
            horiz_delta = horiz_delta - np.size(zf, 1)

        # Map the position to the image space
        pos = pos + self.cell_size * np.array([vert_delta, horiz_delta])

        return pos

    def gaussian_correlation(self, xf, yf):
        N = xf.shape[0] * xf.shape[1]
        xff = xf.reshape([xf.shape[0] * xf.shape[1] * xf.shape[2], 1], order='F')
        xff_T = xff.conj().T
        yff = yf.reshape([yf.shape[0] * yf.shape[1] * yf.shape[2], 1], order='F')
        yff_T = yff.conj().T
        xx = np.dot(xff_T, xff).real / N  # squared norm of x
        yy = np.dot(yff_T, yff).real / N  # squared norm of y
        # cross-correlation term in Fourier domain
        xyf = xf * conj(yf)
        ixyf = ifftn(xyf, axes=(0, 1))
        rxyf = real(ixyf)
        xy = np.sum(rxyf, 2)  # to spatial domain

        # calculate gaussian response for all positions, then go back to the Fourier domain
        sz = xf.shape[0] * xf.shape[1] * xf.shape[2]
        mltp = (xx + yy - 2 * xy) / sz
        crpm = -1 / (self.sigma * self.sigma)
        expe = crpm * np.maximum(0, mltp)
        expx = np.exp(expe)
        kf = fftn(expx, axes=(0, 1))

        #kf2 = self.kernel_correlation(xf, yf)

        return kf




        return kf