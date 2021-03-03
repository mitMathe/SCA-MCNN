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
from keras_layer_normalization import LayerNormalization
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot
from tensorflow.python.keras.layers import Lambda
from keras.models import load_model

OPEN_DATASET = 'ascad'

def LOAD_TRACE(data_type, path, tr_no, pt_st, pt_ed):
        if data_type == 'csv':
            train_data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        elif data_type == 'sft':
            in_fp_sft = open(path, "rb")
            sft_header = sf.read_sft_header(in_fp_sft)
            sft_pt_no = np.uint32(sft_header.nTraceLength / 4)
            sft_tr_no = np.uint32(sft_header.nTraceNum)
    
            train_data = np.fromfile(in_fp_sft, np.float32)
            train_data = train_data.reshape(sft_tr_no, sft_pt_no)
    
            in_fp_sft.close()
    
        elif data_type == 'npy':
            train_data = np.load(path)
    
        return train_data[:tr_no, pt_st:pt_ed + 1]
    
def LOAD_PLAIN(data_type, path):
    if data_type == 'txt':
        temp = np.genfromtxt(path, dtype='|S8')
        plain = [[0 for x in range(temp.shape[1])] for y in range(temp.shape[0])]
    
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                plain[i][j] = int(temp[i][j].decode("utf-8"), 16)
    elif data_type == 'npy':
        plain = np.load(path)
    
    return plain


def rank(predictions, key, targets, ntraces, interval=10):
    ranktime = np.zeros(int(ntraces/interval))
    pred = np.zeros(256)

    idx = np.random.randint(predictions.shape[0], size=ntraces)
    
    for i, p in enumerate(idx):
        for k in range(predictions.shape[1]):
            pred[k] += predictions[p, targets[p, k]]
            
        if i % interval == 0:
            ranked = np.argsort(pred)[::-1]
            ranktime[int(i/interval)] = list(ranked).index(key)
            
    return ranktime


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
            if OPEN_DATASET == 'aes_hd':
                #Computes the hypothesis values
                key_log_prob[k] += prediction[i,AES_SBOX_INV[k^int(att_plt[i,11])]^int(att_plt[i,7])]
            else:
                #Computes the hypothesis values
                key_log_prob[k] += prediction[i,AES_SBOX[k^att_plt[i, byte]]] 
        rank_evol[i] = rk_key(key_log_prob,key[byte])

    return rank_evol

def prob_compute(prediction, att_plt, key, byte):
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
    rank_evol = np.zeros((nb_trs,256)) #np.full(nb_trs,255)
    #prediction = np.log(prediction+1e-40)
                            
    for i in range(nb_trs):
        for k in range(nb_hyp):
            if OPEN_DATASET == 'aes_hd':
                #Computes the hypothesis values
                key_log_prob[k] += prediction[i,AES_SBOX_INV[k^int(att_plt[i,11])]^int(att_plt[i,7])]
            else:
                #Computes the hypothesis values
                key_log_prob[k] += prediction[i,AES_SBOX[k^att_plt[i, byte]]] 
                
        for k in range(nb_hyp):
            rank_evol[i][k] = key_log_prob[k] / (i+1) #rk_key(key_log_prob,key[byte])

    return rank_evol

def perform_probability(nb_traces, predictions, nb_attacks, plt, key, byte=0, shuffle=True, savefig=True, filename='fig'):
    (nb_total, nb_hyp) = predictions.shape
    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    
    att_pred = predictions[:nb_traces]
    att_plt = plt[:nb_traces]
    
    all_prob = prob_compute(att_pred,att_plt,key,byte=byte)
    
    return all_prob

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
            #print(i)
    
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

G_DATA_ROOT_PATH   = "../../SCA_DATA/ASCAD/N0=0"
G_TRAIN_DATA_FILE  = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=0)_profiling_50000tr_700pt.npy"
G_TRAIN_PLAIN_FILE = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=0)_profiling_50000tr_700pt_plain.npy"
G_VALID_DATA_FILE  = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=0)_validation_10000tr_700pt.npy"
G_VALID_PLAIN_FILE = G_DATA_ROOT_PATH + "/" + "ASCAD(N0=0)_validation_10000tr_700pt_plain.npy"

train_data = LOAD_TRACE('npy', G_TRAIN_DATA_FILE, 45000, 0, 699)
train_plain = LOAD_PLAIN('npy', G_TRAIN_PLAIN_FILE)

valid_data = LOAD_TRACE('npy', G_VALID_DATA_FILE, 10000, 0, 699)
valid_plain = LOAD_PLAIN('npy', G_VALID_PLAIN_FILE)

model_org = load_model('BN(org).hdf5')
model_ma  = load_model('BN(ma).hdf5')
model_pca = load_model('BN(pca).hdf5')

correct_key = [0x4D, 0xFB, 0xE0, 0xF2, 0x72, 0x21, 0xFE, 0x10, 0xA7, 0x8D, 0x4A, 0xDC, 0x8E, 0x49, 0x04, 0x69]

train_data = train_data.astype('float32')
valid_data = valid_data.astype('float32')

##### Calculating the moving average
ma_base, ma_step, ma_no = 100, 1, 1
(ma_train, ma_len) = MOVING_AVG(train_data, ma_base, ma_step, ma_no)
(ma_valid, ma_len) = MOVING_AVG(valid_data, ma_base, ma_step, ma_no)
            
# Standardization and Normalzation (between 0 and 1)
scaler = preprocessing.StandardScaler()
ma_train = scaler.fit_transform(ma_train)
ma_valid = scaler.transform(ma_valid)
    
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
ma_train = scaler.fit_transform(ma_train)
ma_valid = scaler.fit_transform(ma_valid)
            
attack_data_ma = ma_valid[5000:10000]
reshape_attack_data_ma = attack_data_ma.reshape((attack_data_ma.shape[0], attack_data_ma.shape[1], 1))


(pc_train, pc_len) = PCA_REDUCTION(train_data)
(pc_valid, pc_len) = PCA_REDUCTION(valid_data)
    
# Standardization and Normalzation (between 0 and 1)
scaler = preprocessing.StandardScaler()
pc_train = scaler.fit_transform(pc_train)
pc_valid = scaler.transform(pc_valid)
    
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
pc_train = scaler.fit_transform(pc_train)
pc_valid = scaler.fit_transform(pc_valid)

attack_data_pca = pc_valid[5000:10000]
reshape_attack_data_pca = attack_data_pca.reshape((attack_data_pca.shape[0], attack_data_pca.shape[1], 1))
        
# Standardization and Normalzation (between 0 and 1)
scaler = preprocessing.StandardScaler()
train_data = scaler.fit_transform(train_data)
valid_data = scaler.transform(valid_data)
        
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(train_data)
valid_data = scaler.fit_transform(valid_data)

attack_data = valid_data[5000:10000]
attack_plain = valid_plain[5000:10000]
reshape_attack_data = attack_data.reshape((attack_data.shape[0], attack_data.shape[1], 1))

attack_inv = np.zeros(5000)
for idx in range(5000):
    attack_inv[idx] = AES_SBOX[int(correct_key[2])^int(attack_plain[idx][2])]
    
attack_prob = np.zeros(256)

for idx in range(5000):
    attack_prob[int(attack_inv[idx])] = attack_prob[int(attack_inv[idx])] + 1
    
for idx in range(256):
    attack_prob[idx] = attack_prob[idx] / 5000

fp = open('ensemble_result.txt', "w")

NO_TRACE = 5000
NO_REPEAT = 100

predictions_org = model_org.predict(reshape_attack_data)
predictions_ma  = model_ma.predict(reshape_attack_data_ma)
predictions_pca = model_pca.predict(reshape_attack_data_pca)

predictions = np.zeros((5000,256))

for idx in range(5000):
    for idx2 in range(256):
        predictions[idx][idx2] = (predictions_org[idx][idx2] + predictions_ma[idx][idx2] +predictions_pca[idx][idx2]) / 3

avg_rank = perform_attacks(NO_TRACE, predictions, NO_REPEAT, plt=attack_plain, key=correct_key, byte=2, shuffle=True, filename='ensemble_result')

for idx in range(avg_rank.shape[0]):
    fp.write("%f " % avg_rank[idx])
fp.write("\n")

fig = plt.figure(figsize=(20, 10))
plt.plot(avg_rank, label=('#Trace 5000'))
plt.rcParams["figure.figsize"] = (20,10)
plt.legend(fontsize='x-large')
plt.show()

