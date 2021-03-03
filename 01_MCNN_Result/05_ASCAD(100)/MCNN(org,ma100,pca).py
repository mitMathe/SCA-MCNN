import os.path
import sys
import h5py
import numpy as np
from numpy import *
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import struct
from ctypes import *
import tensorflow as tf
import os
import time
import shutil
import sys
import binascii
import pickle
from keras.models import Model
from keras.layers import Concatenate, Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import preprocessing
import warnings
from keras.callbacks import Callback
from keras import backend as K
import sklearn
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from operator import itemgetter
from keras.utils import plot_model
from IPython.display import SVG
from tensorflow.python.keras.layers import Lambda

###################################################################
##########################  PARAMETER  ############################
###################################################################
G_IV_PRINT     = False
G_INFO_PRINT   = False
G_RESULT_PRINT = True
G_RESULT_SAVE  = True

# "aes_hd" "ascad100" "ascad50" "ascad0" "aes_rd" "aes_hd_mm"
G_OPEN_DATASET = "ascad100"
# "original" "moving_average" "pca"
G_PREPROCESS = "original"

G_DATA_ROOT_PATH   = "/content/drive/MyDrive/Colab_Data/04_MCNN/AES_HD"
G_TRAIN_DATA_FILE  = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=100)_profiling_50000tr_700pt.npy"
G_TRAIN_PLAIN_FILE = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=100)_profiling_50000tr_700pt_plain.npy"
G_VALID_DATA_FILE  = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=100)_validation_10000tr_700pt.npy"
G_VALID_PLAIN_FILE = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=100)_validation_10000tr_700pt_plain.npy"
G_GEN_RESULT_PATH  = "."

G_TRAIN_NO = 45000
G_VALID_NO = 5000
G_ATTACK_NO = 5000

G_PLAIN_NO = 16
G_BIT_DEPTH = 8
G_OUT_SIZE = 256

G_PT_ST = 0
G_PT_ED = 699
G_LEARN_RATE = 0.01

G_IN_SIZE = G_PT_ED - G_PT_ST + 1
G_LEARN_RATE_ST = G_LEARN_RATE
G_LEARN_RATE_ED = G_LEARN_RATE / 100000

# MASSIVE HYPERPARAMETER
G_EPOCH = 50
G_BATCH = 256
G_LAYER_CNN = 2
G_LAYER = 3
G_LAYER_NO = [20, 20, 20]



class C_SFT_HEADER(Structure):
    _fields_ = [
        ("ucVariable", c_uint8),
        ("ucTypeofTrace", c_uint8),
        ("ucReserved_1", c_uint8),
        ("ucReserved_2", c_uint8),
        ("strID_1", c_int32),
        ("strID_2", c_int32),
        ("nFrequency", c_uint32),
        ("nTraceNum", c_uint32),
        ("nTraceLength", c_uint32),
        ("fOffset", c_float),
        ("fGain", c_float)
    ]

class C_MPL_HYPERPARAMETER(Structure):
    _fields_ = [
        ("learn_rate", c_float),
        ("epoch_size", c_uint32),
        ("batch_size", c_uint32),
        ("layer_size", c_uint32),
        ("p_layer_net_size", POINTER(c_uint32)),
        ("layer_size_cnn", c_uint32),
        ("local_layer_size_cnn", c_uint32),
        ("train_no", c_uint32),
        ("train_size", c_uint32),
        ("valid_no", c_uint32),
        ("valid_size", c_uint32),
        ("attack_no", c_uint32),
        ("in_size", c_uint32),
        ("out_size", c_uint32)
    ]
    
def COPY_HYPER(DST_HYPER, DEP_HYPER):
    DST_HYPER.learn_rate = DEP_HYPER.learn_rate
    DST_HYPER.epoch_size = DEP_HYPER.epoch_size
    DST_HYPER.batch_size = DEP_HYPER.batch_size
    DST_HYPER.layer_size = DEP_HYPER.layer_size

    layer_no = (c_uint32 * DEP_HYPER.layer_size)()
    for i in range(DEP_HYPER.layer_size):
        layer_no[i] = DEP_HYPER.p_layer_net_size[i]
    DST_HYPER.p_lyaer_net_size = layer_no

    DST_HYPER.layer_size_cnn = DEP_HYPER.layer_size_cnn
    DST_HYPER.train_no = DEP_HYPER.train_no
    DST_HYPER.train_size = DEP_HYPER.train_size
    DST_HYPER.valid_no = DEP_HYPER.valid_no
    DST_HYPER.valid_size = DEP_HYPER.valid_size
    DST_HYPER.attack_no = DEP_HYPER.attack_no
    DST_HYPER.in_size = DEP_HYPER.in_size
    DST_HYPER.out_size = DEP_HYPER.out_size

def GET_TODAY():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (
    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def MAKE_FOLDER(folder_name):
    work_dir = G_GEN_RESULT_PATH + "/" + folder_name
    if not os.path.isdir(folder_name):
        os.mkdir(work_dir)
    return work_dir

def DEBUG_PRINT(s, print_on_off):
    if print_on_off:
        print(s)

def SHUFFLE_SCA_DATA(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

def INV_CAL(PLAIN, PLAIN_NO, GUESS_POS, GUESS_VALUE, INTERMEDIATE):
    AES_SBOX = [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
                0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
                0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
                0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
                0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
                0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
                0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
                0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
                0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
                0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
                0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
                0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
                0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
                0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
                0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
                0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]

    AES_SBOX_INV = np.array([0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

    for i in range(PLAIN_NO):
        if G_OPEN_DATASET == 'aes_hd':
            INTERMEDIATE[i] = AES_SBOX_INV[int(PLAIN[i][11]) ^ GUESS_VALUE] ^ int(PLAIN[i][7])
        else:
            INTERMEDIATE[i] = AES_SBOX[PLAIN[i][GUESS_POS] ^ GUESS_VALUE]

def LOAD_TRACE(data_type, path, tr_no, pt_st, pt_ed):
    if data_type == 'npy':
        train_data = np.load(path)
    return train_data[:tr_no, pt_st:pt_ed + 1]

def LOAD_PLAIN(data_type, path):
    if data_type == 'npy':
        plain = np.load(path)
    return plain

# Code implemented by https://github.com/titu1994/keras-one-cycle
# Code is ported from https://github.com/fastai/fastai
class OneCycleLR(Callback):
    def __init__(self,
                max_lr,
                end_percentage=0.1,
                scale_percentage=None,
                maximum_momentum=0.95,
                minimum_momentum=0.85,
                verbose=True):
        """ This callback implements a cyclical learning rate policy (CLR).
        This is a special case of Cyclic Learning Rates, where we have only 1 cycle.
        After the completion of 1 cycle, the learning rate will decrease rapidly to
        100th its initial lowest value.

        # Arguments:
            max_lr: Float. Initial learning rate. This also sets the
                starting learning rate (which will be 10x smaller than
                this), and will increase to this value during the first cycle.
            end_percentage: Float. The percentage of all the epochs of training
                that will be dedicated to sharply decreasing the learning
                rate after the completion of 1 cycle. Must be between 0 and 1.
            scale_percentage: Float or None. If float, must be between 0 and 1.
                If None, it will compute the scale_percentage automatically
                based on the `end_percentage`.
            maximum_momentum: Optional. Sets the maximum momentum (initial)
                value, which gradually drops to its lowest value in half-cycle,
                then gradually increases again to stay constant at this max value.
                Can only be used with SGD Optimizer.
            minimum_momentum: Optional. Sets the minimum momentum at the end of
                the half-cycle. Can only be used with SGD Optimizer.
            verbose: Bool. Whether to print the current learning rate after every
                epoch.

        # Reference
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
            - [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        """
        super(OneCycleLR, self).__init__()

        if end_percentage < 0. or end_percentage > 1.:
            raise ValueError("`end_percentage` must be between 0 and 1")

        if scale_percentage is not None and (scale_percentage < 0. or scale_percentage > 1.):
            raise ValueError("`scale_percentage` must be between 0 and 1")

        self.initial_lr = max_lr
        self.end_percentage = end_percentage
        self.scale = float(scale_percentage) if scale_percentage is not None else float(end_percentage)
        self.max_momentum = maximum_momentum
        self.min_momentum = minimum_momentum
        self.verbose = verbose

        if self.max_momentum is not None and self.min_momentum is not None:
            self._update_momentum = True
        else:
            self._update_momentum = False

        self.clr_iterations = 0.
        self.history = {}

        self.epochs = None
        self.batch_size = None
        self.samples = None
        self.steps = None
        self.num_iterations = None
        self.mid_cycle_id = None

    def _reset(self):
        """
        Reset the callback.
        """
        self.clr_iterations = 0.
        self.history = {}

    def compute_lr(self):
        """
        Compute the learning rate based on which phase of the cycle it is in.

        - If in the first half of training, the learning rate gradually increases.
        - If in the second half of training, the learning rate gradually decreases.
        - If in the final `end_percentage` portion of training, the learning rate
            is quickly reduced to near 100th of the original min learning rate.

        # Returns:
            the new learning rate
        """
        if self.clr_iterations > 2 * self.mid_cycle_id:
            current_percentage = (self.clr_iterations - 2 * self.mid_cycle_id)
            current_percentage /= float((self.num_iterations - 2 * self.mid_cycle_id))
            new_lr = self.initial_lr * (1. + (current_percentage *
                                            (1. - 100.) / 100.)) * self.scale

        elif self.clr_iterations > self.mid_cycle_id:
            current_percentage = 1. - (
                self.clr_iterations - self.mid_cycle_id) / self.mid_cycle_id
            new_lr = self.initial_lr * (1. + current_percentage *
                                        (self.scale * 100 - 1.)) * self.scale

        else:
            current_percentage = self.clr_iterations / self.mid_cycle_id
            new_lr = self.initial_lr * (1. + current_percentage *
                                        (self.scale * 100 - 1.)) * self.scale

        if self.clr_iterations == self.num_iterations:
            self.clr_iterations = 0

        return new_lr

    def compute_momentum(self):
        """
        Compute the momentum based on which phase of the cycle it is in.

        - If in the first half of training, the momentum gradually decreases.
        - If in the second half of training, the momentum gradually increases.
        - If in the final `end_percentage` portion of training, the momentum value
            is kept constant at the maximum initial value.

        # Returns:
            the new momentum value
        """
        if self.clr_iterations > 2 * self.mid_cycle_id:
            new_momentum = self.max_momentum

        elif self.clr_iterations > self.mid_cycle_id:
            current_percentage = 1. - ((self.clr_iterations - self.mid_cycle_id) / float(
                                        self.mid_cycle_id))
            new_momentum = self.max_momentum - current_percentage * (
                self.max_momentum - self.min_momentum)

        else:
            current_percentage = self.clr_iterations / float(self.mid_cycle_id)
            new_momentum = self.max_momentum - current_percentage * (
                self.max_momentum - self.min_momentum)

        return new_momentum

    def on_train_begin(self, logs={}):
        logs = logs or {}

        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size']
        self.samples = self.params['samples']
        self.steps = self.params['steps']

        if self.steps is not None:
            self.num_iterations = self.epochs * self.steps
        else:
            if (self.samples % self.batch_size) == 0:
                remainder = 0
            else:
                remainder = 1
            self.num_iterations = (self.epochs + remainder) * self.samples // self.batch_size

        self.mid_cycle_id = int(self.num_iterations * ((1. - self.end_percentage)) / float(2))

        self._reset()
        K.set_value(self.model.optimizer.lr, self.compute_lr())

        if self._update_momentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be updated only on SGD optimizer !")

            new_momentum = self.compute_momentum()
            K.set_value(self.model.optimizer.momentum, new_momentum)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        self.clr_iterations += 1
        new_lr = self.compute_lr()

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        K.set_value(self.model.optimizer.lr, new_lr)

        if self._update_momentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be updated only on SGD optimizer !")

            new_momentum = self.compute_momentum()

            self.history.setdefault('momentum', []).append(
                K.get_value(self.model.optimizer.momentum))
            K.set_value(self.model.optimizer.momentum, new_momentum)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self._update_momentum:
                print(" - lr: %0.5f - momentum: %0.2f " %
                    (self.history['lr'][-1], self.history['momentum'][-1]))

            else:
                print(" - lr: %0.5f " % (self.history['lr'][-1]))


class LRFinder(Callback):
    def __init__(self,
                num_samples,
                batch_size,
                minimum_lr=1e-5,
                maximum_lr=10.,
                lr_scale='exp',
                validation_data=None,
                validation_sample_rate=5,
                stopping_criterion_factor=4.,
                loss_smoothing_beta=0.98,
                save_dir=None,
                verbose=True):
        """
        This class uses the Cyclic Learning Rate history to find a
        set of learning rates that can be good initializations for the
        One-Cycle training proposed by Leslie Smith in the paper referenced
        below.

        A port of the Fast.ai implementation for Keras.

        # Note
        This requires that the model be trained for exactly 1 epoch. If the model
        is trained for more epochs, then the metric calculations are only done for
        the first epoch.

        # Interpretation
        Upon visualizing the loss plot, check where the loss starts to increase
        rapidly. Choose a learning rate at somewhat prior to the corresponding
        position in the plot for faster convergence. This will be the maximum_lr lr.
        Choose the max value as this value when passing the `max_val` argument
        to OneCycleLR callback.

        Since the plot is in log-scale, you need to compute 10 ^ (-k) of the x-axis

        # Arguments:
            num_samples: Integer. Number of samples in the dataset.
            batch_size: Integer. Batch size during training.
            minimum_lr: Float. Initial learning rate (and the minimum).
            maximum_lr: Float. Final learning rate (and the maximum).
            lr_scale: Can be one of ['exp', 'linear']. Chooses the type of
                scaling for each update to the learning rate during subsequent
                batches. Choose 'exp' for large range and 'linear' for small range.
            validation_data: Requires the validation dataset as a tuple of
                (X, y) belonging to the validation set. If provided, will use the
                validation set to compute the loss metrics. Else uses the training
                batch loss. Will warn if not provided to alert the user.
            validation_sample_rate: Positive or Negative Integer. Number of batches to sample from the
                validation set per iteration of the LRFinder. Larger number of
                samples will reduce the variance but will take longer time to execute
                per batch.

                If Positive > 0, will sample from the validation dataset
                If Megative, will use the entire dataset
            stopping_criterion_factor: Integer or None. A factor which is used
                to measure large increase in the loss value during training.
                Since callbacks cannot stop training of a model, it will simply
                stop logging the additional values from the epochs after this
                stopping criterion has been met.
                If None, this check will not be performed.
            loss_smoothing_beta: Float. The smoothing factor for the moving
                average of the loss function.
            save_dir: Optional, String. If passed a directory path, the callback
                will save the running loss and learning rates to two separate numpy
                arrays inside this directory. If the directory in this path does not
                exist, they will be created.
            verbose: Whether to print the learning rate after every batch of training.

        # References:
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
        """
        super(LRFinder, self).__init__()

        if lr_scale not in ['exp', 'linear']:
            raise ValueError("`lr_scale` must be one of ['exp', 'linear']")

        if validation_data is not None:
            self.validation_data = validation_data
            self.use_validation_set = True

            if validation_sample_rate > 0 or validation_sample_rate < 0:
                self.validation_sample_rate = validation_sample_rate
            else:
                raise ValueError("`validation_sample_rate` must be a positive or negative integer other than o")
        else:
            self.use_validation_set = False
            self.validation_sample_rate = 0

        self.num_samples = num_samples
        self.batch_size = batch_size
        self.initial_lr = minimum_lr
        self.final_lr = maximum_lr
        self.lr_scale = lr_scale
        self.stopping_criterion_factor = stopping_criterion_factor
        self.loss_smoothing_beta = loss_smoothing_beta
        self.save_dir = save_dir
        self.verbose = verbose

        self.num_batches_ = num_samples // batch_size
        self.current_lr_ = minimum_lr

        if lr_scale == 'exp':
            self.lr_multiplier_ = (maximum_lr / float(minimum_lr)) ** (
                1. / float(self.num_batches_))
        else:
            extra_batch = int((num_samples % batch_size) != 0)
            self.lr_multiplier_ = np.linspace(
                minimum_lr, maximum_lr, num=self.num_batches_ + extra_batch)

        # If negative, use entire validation set
        if self.validation_sample_rate < 0:
            self.validation_sample_rate = self.validation_data[0].shape[0] // batch_size

        self.current_batch_ = 0
        self.current_epoch_ = 0
        self.best_loss_ = 1e6
        self.running_loss_ = 0.

        self.history = {}

    def on_train_begin(self, logs=None):

        self.current_epoch_ = 1
        K.set_value(self.model.optimizer.lr, self.initial_lr)

        warnings.simplefilter("ignore")

    def on_epoch_begin(self, epoch, logs=None):
        self.current_batch_ = 0

        if self.current_epoch_ > 1:
            warnings.warn(
                "\n\nLearning rate finder should be used only with a single epoch. "
                "Hereafter, the callback will not measure the losses.\n\n")

    def on_batch_begin(self, batch, logs=None):
        self.current_batch_ += 1

    def on_batch_end(self, batch, logs=None):
        if self.current_epoch_ > 1:
            return

        if self.use_validation_set:
            X, Y = self.validation_data[0], self.validation_data[1]

            # use 5 random batches from test set for fast approximate of loss
            num_samples = self.batch_size * self.validation_sample_rate

            if num_samples > X.shape[0]:
                num_samples = X.shape[0]

            idx = np.random.choice(X.shape[0], num_samples, replace=False)
            x = X[idx]
            y = Y[idx]

            values = self.model.evaluate(x, y, batch_size=self.batch_size, verbose=False)
            loss = values[0]
        else:
            loss = logs['loss']

        # smooth the loss value and bias correct
        running_loss = self.loss_smoothing_beta * loss + (
            1. - self.loss_smoothing_beta) * loss
        running_loss = running_loss / (
            1. - self.loss_smoothing_beta**self.current_batch_)

        # stop logging if loss is too large
        if self.current_batch_ > 1 and self.stopping_criterion_factor is not None and (
                running_loss >
                self.stopping_criterion_factor * self.best_loss_):

            if self.verbose:
                print(" - LRFinder: Skipping iteration since loss is %d times as large as best loss (%0.4f)"
                    % (self.stopping_criterion_factor, self.best_loss_))
            return

        if running_loss < self.best_loss_ or self.current_batch_ == 1:
            self.best_loss_ = running_loss

        current_lr = K.get_value(self.model.optimizer.lr)

        self.history.setdefault('running_loss_', []).append(running_loss)
        if self.lr_scale == 'exp':
            self.history.setdefault('log_lrs', []).append(np.log10(current_lr))
        else:
            self.history.setdefault('log_lrs', []).append(current_lr)

        # compute the lr for the next batch and update the optimizer lr
        if self.lr_scale == 'exp':
            current_lr *= self.lr_multiplier_
        else:
            current_lr = self.lr_multiplier_[self.current_batch_ - 1]

        K.set_value(self.model.optimizer.lr, current_lr)

        # save the other metrics as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.verbose:
            if self.use_validation_set:
                print(" - LRFinder: val_loss: %1.4f - lr = %1.8f " %
                    (values[0], current_lr))
            else:
                print(" - LRFinder: lr = %1.8f " % current_lr)

    def on_epoch_end(self, epoch, logs=None):
        if self.save_dir is not None and self.current_epoch_ <= 1:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            losses_path = os.path.join(self.save_dir, 'losses.npy')
            lrs_path = os.path.join(self.save_dir, 'lrs.npy')

            np.save(losses_path, self.losses)
            np.save(lrs_path, self.lrs)

            if self.verbose:
                print("\tLR Finder : Saved the losses and learning rate values in path : {%s}"
                    % (self.save_dir))

        self.current_epoch_ += 1

        warnings.simplefilter("default")

    def plot_schedule(self, clip_beginning=None, clip_endding=None):
        """
        Plots the schedule from the callback itself.

        # Arguments:
            clip_beginning: Integer or None. If positive integer, it will
                remove the specified portion of the loss graph to remove the large
                loss values in the beginning of the graph.
            clip_endding: Integer or None. If negative integer, it will
                remove the specified portion of the ending of the loss graph to
                remove the sharp increase in the loss values at high learning rates.
        """
        try:
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-white')
        except ImportError:
            print(
                "Matplotlib not found. Please use `pip install matplotlib` first."
            )
            return

        if clip_beginning is not None and clip_beginning < 0:
            clip_beginning = -clip_beginning

        if clip_endding is not None and clip_endding > 0:
            clip_endding = -clip_endding

        losses = self.losses
        lrs = self.lrs

        if clip_beginning:
            losses = losses[clip_beginning:]
            lrs = lrs[clip_beginning:]

        if clip_endding:
            losses = losses[:clip_endding]
            lrs = lrs[:clip_endding]

        plt.plot(lrs, losses)
        plt.title('Learning rate vs Loss')
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.show()

    @classmethod
    def restore_schedule_from_dir(cls,
                                directory,
                                clip_beginning=None,
                                clip_endding=None):
        """
        Loads the training history from the saved numpy files in the given directory.

        # Arguments:
            directory: String. Path to the directory where the serialized numpy
                arrays of the loss and learning rates are saved.
            clip_beginning: Integer or None. If positive integer, it will
                remove the specified portion of the loss graph to remove the large
                loss values in the beginning of the graph.
            clip_endding: Integer or None. If negative integer, it will
                remove the specified portion of the ending of the loss graph to
                remove the sharp increase in the loss values at high learning rates.

        Returns:
            tuple of (losses, learning rates)
        """
        if clip_beginning is not None and clip_beginning < 0:
            clip_beginning = -clip_beginning

        if clip_endding is not None and clip_endding > 0:
            clip_endding = -clip_endding

        losses_path = os.path.join(directory, 'losses.npy')
        lrs_path = os.path.join(directory, 'lrs.npy')

        if not os.path.exists(losses_path) or not os.path.exists(lrs_path):
            print("%s and %s could not be found at directory : {%s}" %
                (losses_path, lrs_path, directory))

            losses = None
            lrs = None

        else:
            losses = np.load(losses_path)
            lrs = np.load(lrs_path)

            if clip_beginning:
                losses = losses[clip_beginning:]
                lrs = lrs[clip_beginning:]

            if clip_endding:
                losses = losses[:clip_endding]
                lrs = lrs[:clip_endding]

        return losses, lrs

    @classmethod
    def plot_schedule_from_file(cls,
                                directory,
                                clip_beginning=None,
                                clip_endding=None):
        """
        Plots the schedule from the saved numpy arrays of the loss and learning
        rate values in the specified directory.

        # Arguments:
            directory: String. Path to the directory where the serialized numpy
                arrays of the loss and learning rates are saved.
            clip_beginning: Integer or None. If positive integer, it will
                remove the specified portion of the loss graph to remove the large
                loss values in the beginning of the graph.
            clip_endding: Integer or None. If negative integer, it will
                remove the specified portion of the ending of the loss graph to
                remove the sharp increase in the loss values at high learning rates.
        """
        try:
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-white')
        except ImportError:
            print("Matplotlib not found. Please use `pip install matplotlib` first.")
            return

        losses, lrs = cls.restore_schedule_from_dir(
            directory,
            clip_beginning=clip_beginning,
            clip_endding=clip_endding)

        if losses is None or lrs is None:
            return
        else:
            plt.plot(lrs, losses)
            plt.title('Learning rate vs Loss')
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.show()

    @property
    def lrs(self):
        return np.array(self.history['log_lrs'])

    @property
    def losses(self):
        return np.array(self.history['running_loss_'])

###################################################################
########################  LOADING DATA  ###########################
###################################################################
AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

AES_SBOX_INV = np.array([0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])


###################################################################
##########################  FUNCTIONS  ############################
###################################################################

# Compute the position of the key hypothesis key amongst the hypotheses
def rk_key(rank_array,key):
    key_val = rank_array[key]
    return np.where(np.sort(rank_array)[::-1] == key_val)[0][0]

# Compute the evolution of rank
def rank_compute(prediction, att_plt, key, byte):
    """
    - prediction : predictions of the NN 
    - att_plt : plaintext of the attack traces 
    - key : Key used during encryption
    - byte : byte to attack
    """
    
    (nb_trs, nb_hyp) = prediction.shape
    
    idx_min = nb_trs
    min_rk = 255
    
    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs,255)
    prediction = np.log(prediction+1e-40)
                            
    for i in range(nb_trs):
        for k in range(nb_hyp):
            if G_OPEN_DATASET == 'aes_hd':
                #Computes the hypothesis values
                key_log_prob[k] += prediction[i,AES_SBOX_INV[k^int(att_plt[i,11])]^int(att_plt[i,7])]
            else:
                #Computes the hypothesis values
                key_log_prob[k] += prediction[i,AES_SBOX[k^att_plt[i, byte]]] 
        rank_evol[i] = rk_key(key_log_prob,key[byte])

    return rank_evol

# Performs attack
def perform_attacks(nb_traces, predictions, nb_attacks, plt, key, byte=0, shuffle=True, savefig=True, filename='fig'):
    """
    Performs a given number of attacks to be determined

    - nb_traces : number of traces used to perform the attack
    - predictions : array containing the values of the prediction
    - nb_attacks : number of attack to perform
    - plt : the plaintext used to obtain the consumption traces
    - key : the key used to obtain the consumption traces
    - byte : byte to attack
    - shuffle (boolean, default = True)

    """

    (nb_total, nb_hyp) = predictions.shape

    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in tqdm(range(nb_attacks)):
        if shuffle:
            l = list(zip(predictions,plt))
            random.shuffle(l)
            sp,splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt[:nb_traces]
        
        rank_evolution = rank_compute(att_pred,att_plt,key,byte=byte)
        all_rk_evol[i] = rank_evolution

    rk_avg = np.mean(all_rk_evol,axis=0)
    
    return (rk_avg)

def MASSIVE_SCA_DL(RUN_FUNCTION=None, BACKUP_FILE=None, DATA_TYPE='npy', GPU_CONFIG=None):
    # Creating the work folder based on current time
    if (G_RESULT_SAVE == 1):
        st_t = time.time()
        work_dir = MAKE_FOLDER(GET_TODAY())

    # Allocation to train data
    train_data = LOAD_TRACE(DATA_TYPE, G_TRAIN_DATA_FILE, G_TRAIN_NO, G_PT_ST, G_PT_ED)
    if DATA_TYPE == 'npy':
        train_plain = LOAD_PLAIN(DATA_TYPE, G_TRAIN_PLAIN_FILE)
    else:
        exit()

    # Allocation to valid data
    valid_data = LOAD_TRACE(DATA_TYPE, G_VALID_DATA_FILE, G_VALID_NO + G_ATTACK_NO, G_PT_ST, G_PT_ED)
    if DATA_TYPE == 'npy':
        valid_plain = LOAD_PLAIN(DATA_TYPE, G_VALID_PLAIN_FILE)
    else:
        exit()

    if G_RESULT_SAVE:
        final_work_file = work_dir + "/" + "final_result.txt"
        fp_r = open(final_work_file, 'w')

    # Generating hyperparameter
    hyperparameter = C_MPL_HYPERPARAMETER()
    
    # Initializing hyperparameter
    hyperparameter.train_no = c_uint32(train_data.shape[0])
    hyperparameter.train_size = c_uint32(train_data.shape[1])
    hyperparameter.valid_no = c_uint32(G_VALID_NO)
    hyperparameter.valid_size = c_uint32(valid_data.shape[1])
    hyperparameter.attack_no = c_uint32(G_ATTACK_NO)
    hyperparameter.out_size = G_OUT_SIZE
    hyperparameter.learn_rate = G_LEARN_RATE
    hyperparameter.in_size = G_IN_SIZE
    
    # Allocating hyperparameter to perform SCA
    layer_no = (c_uint32 * G_LAYER)()
    for i in range(G_LAYER):
        layer_no[i] = 20

    hyperparameter.batch_size = G_BATCH
    hyperparameter.layer_size = G_LAYER
    hyperparameter.epoch_size = G_EPOCH
    hyperparameter.layer_size_cnn = G_LAYER_CNN
    hyperparameter.p_layer_net_size = layer_no
    
    if G_RESULT_SAVE:
        fp_r.write("#####")
        fp_r.write("batch_size: %d, layer_size: %d, epoch_size: %d, " % (hyperparameter.batch_size, hyperparameter.layer_size, hyperparameter.epoch_size))
        fp_r.write("#####")
        
        if G_OPEN_DATASET == 'aes_rd':
            byte = 0
            key = 0x2B
        elif G_OPEN_DATASET == 'aes_hd':
            byte = 0
            key = 0x00
        else:
            byte = 2
            key = 0xE0

        RUN_FUNCTION("log_archive", fp_r, (work_dir + "/"), hyperparameter, byte, key, train_data, train_plain, valid_data, valid_plain, GPU_CONFIG)
    else:
        RUN_FUNCTION("log_archive", "", (work_dir + "/"), hyperparameter, byte, key, train_data, train_plain, valid_data, valid_plain, GPU_CONFIG)

    if G_RESULT_SAVE:
        fp_r.close()
        ed_t = time.time()
        time_file = work_dir + "/elapsed_time.txt"
        fp_t = open(time_file, 'w')
        fp_t.write("elasped time: %f\n" % (ed_t - st_t))
        fp_t.close()
        
def CHES2020_CNN_SCA(LOG_FILE, FP_RESULT, FINAL_PATH, HYPERPARAMETER, GUESS_POS, GUESS_KEY, TRAIN_DATA, TRAIN_PLAIN, VALID_DATA, VALID_PLAIN, GPU_CONFIG):
    if G_OPEN_DATASET == 'aes_rd':
        correct_key = [0x2b, 0x7E, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c] #AES_RD
    elif G_OPEN_DATASET == 'aes_hd':
        correct_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00] #AES_HD
    else:
        correct_key = [0x4D, 0xFB, 0xE0, 0xF2, 0x72, 0x21, 0xFE, 0x10, 0xA7, 0x8D, 0x4A, 0xDC, 0x8E, 0x49, 0x04, 0x69] #ASCAD
    
    train_data = TRAIN_DATA.astype('float32')
    valid_data = VALID_DATA.astype('float32')
    
    # Standardization and Normalzation (between 0 and 1)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.fit_transform(valid_data)
    
    (train_data, TRAIN_PLAIN) = SHUFFLE_SCA_DATA(train_data, TRAIN_PLAIN)

    if G_PREPROCESS == 'original':
        TRAIN_DATA = train_data
        VALID_DATA = valid_data
    elif G_PREPROCESS == 'moving_average':
        ###### Calculating the moving average
        ma_base, ma_step, ma_no = 100, 1, 1
        (ma_train, ma_len) = MOVING_AVG(TRAIN_DATA, ma_base, ma_step, ma_no)
        (ma_valid, ma_len) = MOVING_AVG(VALID_DATA, ma_base, ma_step, ma_no)
        
        # Standardization and Normalzation (between 0 and 1)
        scaler = preprocessing.StandardScaler()
        ma_train = scaler.fit_transform(ma_train)
        ma_valid = scaler.transform(ma_valid)

        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        ma_train = scaler.fit_transform(ma_train)
        ma_valid = scaler.fit_transform(ma_valid)
        
        TRAIN_DATA = ma_train
        VALID_DATA = ma_valid
        HYPERPARAMETER.in_size = ma_train.shape[1]
        HYPERPARAMETER.train_size = ma_train.shape[1]
        HYPERPARAMETER.valid_size = ma_valid.shape[1]
    elif G_PREPROCESS == 'pca':
        ###### Calculating the pca
        (pc_train, pc_len) = PCA_REDUCTION(TRAIN_DATA)
        (pc_valid, pc_len) = PCA_REDUCTION(VALID_DATA)

        # Standardization and Normalzation (between 0 and 1)
        scaler = preprocessing.StandardScaler()
        pc_train = scaler.fit_transform(pc_train)
        pc_valid = scaler.transform(pc_valid)

        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        pc_train = scaler.fit_transform(pc_train)
        pc_valid = scaler.fit_transform(pc_valid)
        
        TRAIN_DATA = pc_train
        VALID_DATA = pc_valid
        HYPERPARAMETER.in_size = pc_train.shape[1]
        HYPERPARAMETER.train_size = pc_train.shape[1]
        HYPERPARAMETER.valid_size = pc_valid.shape[1]
    else:
        print("Type is wrong")
        exit()
        
    valid_data = VALID_DATA[:HYPERPARAMETER.valid_no]
    valid_plain = VALID_PLAIN[:HYPERPARAMETER.valid_no]
    attack_data = VALID_DATA[HYPERPARAMETER.valid_no:HYPERPARAMETER.valid_no+HYPERPARAMETER.attack_no]
    attack_plain = VALID_PLAIN[HYPERPARAMETER.valid_no:HYPERPARAMETER.valid_no+HYPERPARAMETER.attack_no]
    reshape_valid_data = valid_data.reshape((valid_data.shape[0], valid_data.shape[1], 1))
    reshape_attack_data = attack_data.reshape((attack_data.shape[0], attack_data.shape[1], 1))

    model = CHES2020_CNN_ARCHI(HYPERPARAMETER)
    model_name = G_OPEN_DATASET
    print("Model Name = " + model_name)
    print(model.summary())

    st_t = time.time()
    history = CHES2020_CNN_TRAIN(LOG_FILE, FP_RESULT, HYPERPARAMETER, GUESS_POS, GUESS_KEY, TRAIN_DATA, TRAIN_PLAIN, valid_data, valid_plain, GPU_CONFIG, model)
    ed_t = time.time()
    time_file = FINAL_PATH + "train_time.txt"
    fp_t = open(time_file, 'w')
    fp_t.write("elasped time: %f\n" % (ed_t - st_t))
    fp_t.close()
    
    predictions = model.predict(reshape_attack_data)
    if True:
        for layer in model.layers:
            inv_layer = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
            inv_out = inv_layer.predict(reshape_attack_data)
            avg = [0] * inv_out.shape[1]
            if inv_out.ndim == 3:
                for idx2 in range(inv_out.shape[2]):
                    for idx1 in range(inv_out.shape[1]):
                        avg[idx1] += inv_out[0][idx1][idx2]
                for idx1 in range(inv_out.shape[1]):
                    avg[idx1] /= inv_out.shape[2]
            else:
                for idx1 in range(inv_out.shape[1]):
                    avg[idx1] = inv_out[0][idx1]
                    
            INV_PATH = (FINAL_PATH + '%s' + '.npy') % (layer.name)
            np.save(INV_PATH, avg)
                    
            fig = plt.figure(figsize=(20, 10))
            plt.rcParams["figure.figsize"] = (20,10)
            plt.title(layer.name)            
            plt.plot(avg)
            plt.show()
            
            FIG_PATH = (FINAL_PATH + '%s' + '.png') % (layer.name)
            fig.savefig(FIG_PATH, dpi=fig.dpi, bbox_inches="tight")
    
    st_t = time.time()
    avg_rank = perform_attacks(HYPERPARAMETER.attack_no, predictions, 100, plt=attack_plain, key=correct_key, byte=GUESS_POS, filename=model_name)
    ed_t = time.time()
    time_file = FINAL_PATH + "attack_time.txt"
    fp_t = open(time_file, 'w')
    fp_t.write("elasped time: %f\n" % (ed_t - st_t))
    fp_t.close()
    
    print("\n t_GE = ")
    print(avg_rank)
    print(np.where(avg_rank<=0))
    
    if G_RESULT_SAVE:
        for idx in range(avg_rank.shape[0]):
            FP_RESULT.write("%f " % avg_rank[idx])
    
        FP_RESULT.write("\n")
        FP_RESULT.write("%d" % TRAIN_DATA.shape[0])
            
        INV_PATH = (FINAL_PATH + 'GE_result' + '.npy')
        np.save(INV_PATH, avg_rank)
        fig = plt.figure(figsize=(20, 10))
        plt.plot(avg_rank, label=(G_PREPROCESS + ' Result against ' + G_OPEN_DATASET))
        plt.rcParams["figure.figsize"] = (20,10)
        plt.legend(fontsize='x-large')
        FIG_PATH = (FINAL_PATH + 'GE_result' + '.png')
        fig.savefig(FIG_PATH, dpi=fig.dpi, bbox_inches="tight")
        plt.show()
        
        trace = np.load(FINAL_PATH + 'GE_result' + ".npy") 
        plt.plot(trace)
        plt.rcParams["figure.figsize"] = (20,10)
        plt.show()
        
        model.save((FINAL_PATH + 'ORIGINAL_RESULT' + '.hdf5'))

def CHES2020_CNN_ARCHI(HYPERPARAMETER):
    input_shape = (HYPERPARAMETER.in_size, 1)
    img_input = Input(shape=input_shape)
    BN_IDX = [1] * HYPERPARAMETER.layer_size_cnn
    COV_NO_FILTER   = [32, 64, 128]
    COV_SIZE_FILTER = [1, 50, 3]
    POOL_FILTER     = [2, 50, 2]
    LAYER_NO        = [20, 20, 20]

    x = img_input
    for array_idx in range(HYPERPARAMETER.layer_size_cnn):
        x = Conv1D(COV_NO_FILTER[array_idx], COV_SIZE_FILTER[array_idx], kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv%d' % array_idx)(x)
        if BN_IDX[array_idx] == 1:
            x = BatchNormalization()(x)
        x = AveragePooling1D(POOL_FILTER[array_idx], strides=POOL_FILTER[array_idx], name='block%d_pool' % array_idx)(x)
    x = Flatten(name='flatten')(x)
    for array_idx in range(HYPERPARAMETER.layer_size):
        x = Dense(LAYER_NO[array_idx], kernel_initializer='he_uniform', activation='selu', name='fc%d' % array_idx)(x)
    # Logits layer
    x = Dense(HYPERPARAMETER.out_size, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name=G_OPEN_DATASET)
    optimizer = Adam(lr=HYPERPARAMETER.learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def CHES2020_CNN_TRAIN(LOG_FILE, FP_RESULT, HYPERPARAMETER, GUESS_POS, GUESS_KEY, TRAIN_DATA, TRAIN_PLAIN, VALID_DATA, VALID_PLAIN, GPU_CONFIG, MODEL):
    # Save model every epoch
    save_model = ModelCheckpoint(LOG_FILE)

    train_inv = [0] * HYPERPARAMETER.train_no
    INV_CAL(TRAIN_PLAIN, HYPERPARAMETER.train_no, GUESS_POS, GUESS_KEY, train_inv)

    # Calculating the intermediate variables
    train_inv_np = np.array(train_inv)
    train_inv_np = reshape(train_inv_np, (HYPERPARAMETER.train_no, 1))
    
    valid_inv = [0] * HYPERPARAMETER.valid_no
    INV_CAL(VALID_PLAIN, HYPERPARAMETER.valid_no, GUESS_POS, GUESS_KEY, valid_inv)
    
    valid_inv_np = np.array(valid_inv)
    valid_inv_np = reshape(valid_inv_np, (HYPERPARAMETER.valid_no, 1))

    # Get the input layer shape
    input_layer_shape = MODEL.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(TRAIN_DATA[0]):
        print("Input layer error")
        sys.exit(-1)

    # Reshape the train and valid data
    reshape_train_data = TRAIN_DATA.reshape((TRAIN_DATA.shape[0], TRAIN_DATA.shape[1], 1))
    reshape_valid_data = VALID_DATA.reshape((VALID_DATA.shape[0], VALID_DATA.shape[1], 1))

    lr_manager = OneCycleLR(max_lr=HYPERPARAMETER.learn_rate, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None,verbose=True)
    callbacks = [save_model, lr_manager]
    history = MODEL.fit(x=reshape_train_data, y=to_categorical(train_inv_np, num_classes=HYPERPARAMETER.out_size), validation_data=(reshape_valid_data, to_categorical(valid_inv_np, num_classes=HYPERPARAMETER.out_size)), batch_size=HYPERPARAMETER.batch_size, verbose = 1, epochs=HYPERPARAMETER.epoch_size, callbacks=callbacks)

    return history

def MOVING_AVG_SUB(DATA_X, WINDOW_SIZE):
    no = DATA_X.shape[0]
    len = DATA_X.shape[1]
    out_len = len - WINDOW_SIZE + 1
    output = np.zeros((no, out_len))
    for i in range(out_len):
        output[:,i]=np.mean(DATA_X[:,i : i + WINDOW_SIZE], axis=1)

    return output

def MOVING_AVG(DATA_X, WINDOW_BASE, STEP_SIZE, NO):
    if NO == 0:
        return (None, [])
    out = MOVING_AVG_SUB(DATA_X, WINDOW_BASE)
    data_len = [out.shape[1]]
    for i in range(1, NO):
        window_size = WINDOW_BASE + STEP_SIZE * i
        if window_size > DATA_X.shape[1]:
            continue
        new_series = MOVING_AVG_SUB(DATA_X, window_size)
        data_len.append(new_series.shape[1])
        out = np.concatenate([out, new_series], axis=1)
    return (out, data_len)

def SCA_PCA(IN_TRAIN):
    pca_result = PCA(n_components=20)
    return pca_result.fit_transform(IN_TRAIN)

def PCA_REDUCTION(DATA_X):
    pca_data = SCA_PCA(DATA_X)
    return (pca_data, [pca_data.shape[1]])

def MCNN_SCA(LOG_FILE, FP_RESULT, FINAL_PATH, HYPERPARAMETER, GUESS_POS, GUESS_KEY, TRAIN_DATA, TRAIN_PLAIN, VALID_DATA, VALID_PLAIN, GPU_CONFIG):
    if G_OPEN_DATASET == 'aes_rd':
        correct_key = [0x2b, 0x7E, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c] #AES_RD
    elif G_OPEN_DATASET == 'aes_hd':
        correct_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00] #AES_HD
    else:
        correct_key = [0x4D, 0xFB, 0xE0, 0xF2, 0x72, 0x21, 0xFE, 0x10, 0xA7, 0x8D, 0x4A, 0xDC, 0x8E, 0x49, 0x04, 0x69] #ASCAD

    # Generating hyperparameter
    hyperparameter_1 = C_MPL_HYPERPARAMETER()
    hyperparameter_2 = C_MPL_HYPERPARAMETER()
    hyperparameter_3 = C_MPL_HYPERPARAMETER()

    COPY_HYPER(hyperparameter_1, HYPERPARAMETER)
    COPY_HYPER(hyperparameter_2, HYPERPARAMETER)
    COPY_HYPER(hyperparameter_3, HYPERPARAMETER)

    TRAIN_DATA = TRAIN_DATA.astype('float32')
    VALID_DATA = VALID_DATA.astype('float32')
    
    # Calculation to the intermediate variables for train
    train_inv = [0] * HYPERPARAMETER.train_no
    INV_CAL(TRAIN_PLAIN, HYPERPARAMETER.train_no, GUESS_POS, GUESS_KEY, train_inv)

    train_inv_np = np.array(train_inv)
    train_inv_np = reshape(train_inv_np, (HYPERPARAMETER.train_no, 1))

    # Calculation to the intermediate variables for valid
    valid_inv = [0] * (HYPERPARAMETER.valid_no + HYPERPARAMETER.attack_no)
    INV_CAL(VALID_PLAIN, HYPERPARAMETER.valid_no, GUESS_POS, GUESS_KEY, valid_inv)

    valid_inv_np = np.array(valid_inv)
    valid_inv_np = reshape(valid_inv_np, ((HYPERPARAMETER.valid_no+ HYPERPARAMETER.attack_no), 1))

    # Standardization and Normalzation (between 0 and 1)
    scaler = preprocessing.StandardScaler()
    TRAIN_DATA = scaler.fit_transform(TRAIN_DATA)
    VALID_DATA = scaler.transform(VALID_DATA)

    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    TRAIN_DATA = scaler.fit_transform(TRAIN_DATA)
    VALID_DATA = scaler.fit_transform(VALID_DATA)

    (TRAIN_DATA, TRAIN_PLAIN) = SHUFFLE_SCA_DATA(TRAIN_DATA, TRAIN_PLAIN)

    train_data_1 = TRAIN_DATA
    reshape_valid_data_1 = VALID_DATA.reshape((VALID_DATA.shape[0], VALID_DATA.shape[1], 1))
    
    ###### The setting for second CNN Layer
    ma_base, ma_step, ma_no = 100, 1, 1
    (ma_train, ma_len) = MOVING_AVG(TRAIN_DATA, ma_base, ma_step, ma_no)
    (ma_valid, ma_len) = MOVING_AVG(VALID_DATA, ma_base, ma_step, ma_no)

    # Standardization and Normalzation (between 0 and 1)
    scaler = preprocessing.StandardScaler()
    ma_train = scaler.fit_transform(ma_train)
    ma_valid = scaler.transform(ma_valid)

    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    ma_train = scaler.fit_transform(ma_train)
    ma_valid = scaler.fit_transform(ma_valid)
    
    train_data_2 = ma_train
    reshape_valid_data_2 = ma_valid.reshape((ma_valid.shape[0], ma_valid.shape[1], 1))

    hyperparameter_2.in_size = ma_train.shape[1]
    hyperparameter_2.train_size = ma_train.shape[1]
    hyperparameter_2.valid_size = ma_valid.shape[1]

    ###### The setting for third CNN Layer
    (pc_train, pc_len) = PCA_REDUCTION(TRAIN_DATA)
    (pc_valid, pc_len) = PCA_REDUCTION(VALID_DATA)

    # Standardization and Normalzation (between 0 and 1)
    scaler = preprocessing.StandardScaler()
    pc_train = scaler.fit_transform(pc_train)
    pc_valid = scaler.transform(pc_valid)

    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    pc_train = scaler.fit_transform(pc_train)
    pc_valid = scaler.fit_transform(pc_valid)

    hyperparameter_3.in_size = pc_train.shape[1]
    hyperparameter_3.train_size = pc_train.shape[1]
    hyperparameter_3.valid_size = pc_valid.shape[1]

    train_data_3 = pc_train
    reshape_valid_data_3 = pc_valid.reshape((pc_valid.shape[0], pc_valid.shape[1], 1))

    # Split to validation data and attack data
    valid_data_1 = VALID_DATA[:HYPERPARAMETER.valid_no]
    valid_data_2 = ma_valid[:HYPERPARAMETER.valid_no]
    valid_data_3 = pc_valid[:HYPERPARAMETER.valid_no]
    valid_plain = VALID_PLAIN[:HYPERPARAMETER.valid_no]
    
    attack_data_1 = VALID_DATA[HYPERPARAMETER.valid_no:HYPERPARAMETER.valid_no+HYPERPARAMETER.attack_no]
    attack_data_2 = ma_valid[HYPERPARAMETER.valid_no:HYPERPARAMETER.valid_no+HYPERPARAMETER.attack_no]
    attack_data_3 = pc_valid[HYPERPARAMETER.valid_no:HYPERPARAMETER.valid_no+HYPERPARAMETER.attack_no]
    attack_plain = VALID_PLAIN[HYPERPARAMETER.valid_no:HYPERPARAMETER.valid_no+HYPERPARAMETER.attack_no]

    reshape_valid_data_1 = valid_data_1.reshape((valid_data_1.shape[0], valid_data_1.shape[1], 1))
    reshape_valid_data_2 = valid_data_2.reshape((valid_data_2.shape[0], valid_data_2.shape[1], 1))
    reshape_valid_data_3 = valid_data_3.reshape((valid_data_3.shape[0], valid_data_3.shape[1], 1))
    reshape_attack_data_1 = attack_data_1.reshape((attack_data_1.shape[0], attack_data_1.shape[1], 1))
    reshape_attack_data_2 = attack_data_2.reshape((attack_data_2.shape[0], attack_data_2.shape[1], 1))
    reshape_attack_data_3 = attack_data_3.reshape((attack_data_3.shape[0], attack_data_3.shape[1], 1))
    
    model = MCNN_ARCHI(hyperparameter_1, hyperparameter_2, hyperparameter_3)
    model_name = 'MCNN_' + G_OPEN_DATASET
    print("Model Name = " + model_name)
    print(model.summary())
    
    st_t = time.time()
    history = MCNN_TRAIN(LOG_FILE, FP_RESULT, HYPERPARAMETER, GUESS_POS, GUESS_KEY, train_data_1, train_data_2, train_data_3, TRAIN_PLAIN, reshape_valid_data_1, reshape_valid_data_2, reshape_valid_data_3, valid_plain, GPU_CONFIG, model)
    ed_t = time.time()
    time_file = FINAL_PATH + "train_time.txt"
    fp_t = open(time_file, 'w')
    fp_t.write("elasped time: %f\n" % (ed_t - st_t))
    fp_t.close()
    
    predictions = model.predict([reshape_attack_data_1, reshape_attack_data_2, reshape_attack_data_3])
    
    if True:
        for layer in model.layers:
            inv_layer = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
            inv_out = inv_layer.predict([reshape_attack_data_1, reshape_attack_data_2, reshape_attack_data_3])
            avg = [0] * inv_out.shape[1]
            if inv_out.ndim == 3:
                for idx2 in range(inv_out.shape[2]):
                    for idx1 in range(inv_out.shape[1]):
                        avg[idx1] += inv_out[0][idx1][idx2]
                for idx1 in range(inv_out.shape[1]):
                    avg[idx1] /= inv_out.shape[2]
            else:
                for idx1 in range(inv_out.shape[1]):
                    avg[idx1] = inv_out[0][idx1]
                    
            INV_PATH = (FINAL_PATH + '%s' + '.npy') % (layer.name)
            np.save(INV_PATH, avg)
                    
            fig = plt.figure(figsize=(20, 10))
            plt.rcParams["figure.figsize"] = (20,10)
            plt.title(layer.name)            
            plt.plot(avg)
            plt.show()
            
            FIG_PATH = (FINAL_PATH + '%s' + '.png') % (layer.name)
            fig.savefig(FIG_PATH, dpi=fig.dpi, bbox_inches="tight")
    
    st_t = time.time()
    avg_rank = perform_attacks(HYPERPARAMETER.attack_no, predictions, 100, plt=attack_plain, key=correct_key, byte=GUESS_POS, filename=model_name)
    ed_t = time.time()
    time_file = FINAL_PATH + "attack_time.txt"
    fp_t = open(time_file, 'w')
    fp_t.write("elasped time: %f\n" % (ed_t - st_t))
    fp_t.close()
    
    print("\n t_GE = ")
    print(avg_rank)
    print(np.where(avg_rank<=0))
    
    if G_RESULT_SAVE:
        for idx in range(avg_rank.shape[0]):
            FP_RESULT.write("%f " % avg_rank[idx])
    
        FP_RESULT.write("\n")
        FP_RESULT.write("%d" % TRAIN_DATA.shape[0])
            
        INV_PATH = (FINAL_PATH + 'GE_result' + '.npy')
        np.save(INV_PATH, avg_rank)
        fig = plt.figure(figsize=(20, 10))
        plt.plot(avg_rank, label=('MCNN Result against ') + G_OPEN_DATASET)
        plt.rcParams["figure.figsize"] = (20,10)
        plt.legend(fontsize='x-large')
        FIG_PATH = (FINAL_PATH + 'GE_result' + '.png')
        fig.savefig(FIG_PATH, dpi=fig.dpi, bbox_inches="tight")
        plt.show()
        
        trace = np.load(FINAL_PATH + 'GE_result' + ".npy") 
        plt.plot(trace)
        plt.rcParams["figure.figsize"] = (20,10)
        plt.show()
        
        model.save((FINAL_PATH + 'MCNN_RESULT' + '.hdf5'))
    
def MCNN_ARCHI(HYPERPARAMETER_PRE_1, HYPERPARAMETER_PRE_2, HYPERPARAMETER_PRE_3):
    HYPERPARAMETER_PRE_1.layer_size_cnn = 2
    HYPERPARAMETER_PRE_2.layer_size_cnn = 2
    HYPERPARAMETER_PRE_3.layer_size_cnn = 2
    
    in_1 = (HYPERPARAMETER_PRE_1.in_size, 1)
    ig_1 = Input(shape=in_1)

    in_2 = (HYPERPARAMETER_PRE_2.in_size, 1)
    ig_2 = Input(shape=in_2)

    in_3 = (HYPERPARAMETER_PRE_3.in_size, 1)
    ig_3 = Input(shape=in_3)

    COV_NO_1 = [32, 64]
    COV_SZ_1 = [1, 50]
    PL_FIL_1 = [2, 50]

    COV_NO_2 = [32, 64]
    COV_SZ_2 = [1, 50]
    PL_FIL_2 = [2, 50]

    COV_NO_3 = [32, 64]
    COV_SZ_3 = [1, 1]
    PL_FIL_3 = [2, 1]

    COV_NO = [128]
    COV_SZ = [3]
    PL_FIL = [2]

    LAY_NO = [20, 20, 20]
    
    x1 = ig_1
    for array_idx in range(HYPERPARAMETER_PRE_1.layer_size_cnn):
        x1 = Conv1D(COV_NO_1[array_idx], COV_SZ_1[array_idx], kernel_initializer='he_uniform', activation='selu', padding='same', name='first_block%d_conv' % array_idx)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling1D(PL_FIL_1[array_idx], strides=PL_FIL_1[array_idx], name='first_block%d_pool' % array_idx)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Model(inputs=ig_1, outputs=x1)

    x2 = ig_2
    for array_idx in range(HYPERPARAMETER_PRE_2.layer_size_cnn):
        x2 = Conv1D(COV_NO_2[array_idx], COV_SZ_2[array_idx], kernel_initializer='he_uniform', activation='selu', padding='same', name='second_block%d_conv' % array_idx)(x2)
        x2 = BatchNormalization()(x2)
        x2 = AveragePooling1D(PL_FIL_2[array_idx], strides=PL_FIL_2[array_idx], name='second_block%d_pool' % array_idx)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Model(inputs=ig_2, outputs=x2)
    
    x3 = ig_3
    for array_idx in range(HYPERPARAMETER_PRE_3.layer_size_cnn):
        x3 = Conv1D(COV_NO_3[array_idx], COV_SZ_3[array_idx], kernel_initializer='he_uniform', activation='selu', padding='same', name='third_block%d_conv' % array_idx)(x3)
        x3 = BatchNormalization()(x3)
        x3 = AveragePooling1D(PL_FIL_3[array_idx], strides=PL_FIL_3[array_idx], name='third__block%d_pool' % array_idx)(x3)  
    x3 = BatchNormalization()(x3)
    x3 = Model(inputs=ig_3, outputs=x3)
    
    x4 = Concatenate(axis=1)([x1.output, x2.output, x3.output])
    x4 = BatchNormalization()(x4)
    
    for array_idx in range(1):
        x4 = Conv1D(COV_NO[array_idx], COV_SZ[array_idx], kernel_initializer='he_uniform', activation='selu', padding='same', name='fourth_block%d_conv' % array_idx)(x4)
        x4 = BatchNormalization()(x4)
        x4 = AveragePooling1D(PL_FIL[array_idx], strides=PL_FIL[array_idx], name='fourth_block%d_pool' % array_idx)(x4)
    x4 = Flatten(name='flatten_4')(x4)
    
    for array_idx in range(HYPERPARAMETER_PRE_1.layer_size):
        x4 = Dense(LAY_NO[array_idx], kernel_initializer='he_uniform', activation='selu', name='fc%d' % array_idx)(x4)

    # Logits layer
    x4 = Dense(HYPERPARAMETER_PRE_1.out_size, activation='softmax', name='predictions')(x4)

    # Create model
    model = Model(inputs=[x1.input, x2.input, x3.input], outputs=x4, name='mcnn')
    optimizer = Adam(lr=HYPERPARAMETER_PRE_1.learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def MCNN_TRAIN(LOG_FILE, FP_RESULT, HYPERPARAMETER, GUESS_POS, GUESS_KEY, TRAIN_DATA_1, TRAIN_DATA_2, TRAIN_DATA_3, TRAIN_PLAIN, VALID_DATA_1, VALID_DATA_2, VALID_DATA_3, VALID_PLAIN, GPU_CONFIG, MODEL):
    # Save model every epoch
    save_model = ModelCheckpoint(LOG_FILE)

    # Calculation to the intermediate variables for train
    train_inv = [0] * HYPERPARAMETER.train_no
    INV_CAL(TRAIN_PLAIN, HYPERPARAMETER.train_no, GUESS_POS, GUESS_KEY, train_inv)

    train_inv = train_inv[:HYPERPARAMETER.train_no]
    
    train_inv_np = np.array(train_inv)
    train_inv_np = reshape(train_inv_np, (HYPERPARAMETER.train_no, 1))

    # Calculation to the intermediate variables for valid
    valid_inv = [0] * HYPERPARAMETER.valid_no
    INV_CAL(VALID_PLAIN, HYPERPARAMETER.valid_no, GUESS_POS, GUESS_KEY, valid_inv)

    valid_inv = valid_inv[:HYPERPARAMETER.valid_no]
    
    valid_inv_np = np.array(valid_inv)
    valid_inv_np = reshape(valid_inv_np, (HYPERPARAMETER.valid_no, 1))

    # Conver to 3-dimensional shape
    reshape_train_data_1 = TRAIN_DATA_1.reshape((TRAIN_DATA_1.shape[0], TRAIN_DATA_1.shape[1], 1))
    reshape_valid_data_1 = VALID_DATA_1.reshape((VALID_DATA_1.shape[0], VALID_DATA_1.shape[1], 1))

    reshape_train_data_2 = TRAIN_DATA_2.reshape((TRAIN_DATA_2.shape[0], TRAIN_DATA_2.shape[1], 1))
    reshape_valid_data_2 = VALID_DATA_2.reshape((VALID_DATA_2.shape[0], VALID_DATA_2.shape[1], 1))

    reshape_train_data_3 = TRAIN_DATA_3.reshape((TRAIN_DATA_3.shape[0], TRAIN_DATA_3.shape[1], 1))
    reshape_valid_data_3 = VALID_DATA_3.reshape((VALID_DATA_3.shape[0], VALID_DATA_3.shape[1], 1))

    lr_manager = OneCycleLR(max_lr=HYPERPARAMETER.learn_rate, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None,verbose=True)
    callbacks = [save_model, lr_manager]
    history = MODEL.fit(x=[reshape_train_data_1, reshape_train_data_2, reshape_train_data_3], y=to_categorical(train_inv_np, num_classes=HYPERPARAMETER.out_size), validation_data=([reshape_valid_data_1, reshape_valid_data_2, reshape_valid_data_3], to_categorical(valid_inv_np, num_classes=HYPERPARAMETER.out_size)), batch_size=HYPERPARAMETER.batch_size, verbose = 1, epochs=HYPERPARAMETER.epoch_size, callbacks=callbacks)

    return history

#########################################
############## MAIN SOURCE ##############
#########################################
# CHES2020_CNN_SCA
# MCNN_SCA
MASSIVE_SCA_DL(RUN_FUNCTION=MCNN_SCA, BACKUP_FILE=None, DATA_TYPE='npy', GPU_CONFIG=None)