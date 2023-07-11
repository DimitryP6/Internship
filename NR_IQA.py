# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:29:19 2023

@author: ddude
"""

import numpy as np
import time
from scipy import signal
import os
from sklearn.linear_model import LinearRegression
st = time.time()
"""
BLUR
"""
import pydicom

import matplotlib.pyplot as plt


def show_ct_dicom(path):
    #convert from houndsfield to pixel
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array   
    bound = int((np.shape(data)[0]-(np.shape(data)[0]/np.sqrt(2)))/2)  
    data = data[(bound-2):-(bound+2), (bound-2):-(bound+2)]
    n = 0
    while n<4:
        top = (data[0]<-100) | (3000<data[0])
        s = np.sum(top)
        #less than 5 percent invalid
        if s<(int(0.05*data[0].size)):
            n+=1
            data = np.rot90(data)
        # if 5 percent or more invalid take off top row
        else:
            data = np.delete(data, 0, 0)
            n=0
    data = data - dicom.RescaleIntercept
    data = data / dicom.RescaleSlope
    data = (data * 255).astype(np.uint8)
    return data



def Blur_Statistics(img):
    harr = np.array([[0, 0, 0],[-1, 0, 1], [0, 0, 0]])
    varr = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    D_h = abs(signal.convolve2d(img, harr))
    D_v = abs(signal.convolve2d(img, varr))
    
    D_hm = np.average(D_h)
    D_vm = np.average(D_v)
    
    C_h = np.where(D_h>D_hm, D_h, 0)
    C_v = np.where(D_v>D_vm, D_v, 0)
    
    E_h = np.zeros_like(C_h)
    E_v = np.zeros_like(C_v)
    E_h[2:-2, 2:-2] = ((C_h[2:-2, 2:-2]>C_h[2:-2, 1:-3]) & (C_h[2:-2, 2:-2]>C_h[2:-2, 3:-1]))
    E_v[2:-2, 2:-2] = ((C_h[2:-2, 2:-2]>C_h[1:-3, 2:-2]) & (C_h[2:-2, 2:-2]>C_h[3:-1, 2:-2]))
    E_h = E_h[2:-2, 2:-2]
    E_v = E_v[2:-2, 2:-2]
    
    Th_b = 0.1
    
    D_h = D_h[2:-2, 2:-2]
    D_v = D_v[2:-2, 2:-2]
    A_h = D_h/2
    A_v = D_v/2
    
    data = np.copy(img)
    data = data[1:-1, 1:-1]
    
    Br_h = np.where(A_h != 0, (abs(data-A_h))/A_h, Th_b)
    Br_v = np.where(A_v != 0, (abs(data-A_v))/A_v, Th_b)
    BR = np.maximum(Br_h, Br_v)
    
    
    
    blur_sum = np.sum(np.where(BR < Th_b, BR , 0))
    blur_count = np.sum(BR < Th_b) 
    edge_count = np.sum(E_h) + np.sum(E_v == E_h, where = 1)
    #Blur Mean, Blur Ratio
    return(blur_sum/blur_count if blur_count != 0 else 0, blur_count/edge_count if edge_count != 0 else 0)

"""
NOISE
"""

def Noise_Statistics(img):
    
    farr = ([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    fimg = signal.convolve2d(img, farr)/9
    
    harr = np.array([[0, 0, 0],[-1, 0, 1], [0, 0, 0]])
    varr = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    D_h = abs(signal.convolve2d(fimg, harr))
    D_v = abs(signal.convolve2d(fimg, varr))  
    
    D_hm = np.average(D_h)
    D_vm = np.average(D_v)
    
    N_cand = np.where((D_h<=D_hm) & (D_v<=D_vm), np.maximum(D_h, D_v) , 0)
    N_candm = np.average(N_cand)
    
    N = np.where(N_cand>N_candm, N_cand, 0)
    noise_sum = np.average(N)
    noise_count = np.sum(N_cand>N_candm)
    #noise mean and noise ratio
    return (noise_sum/noise_count if noise_count != 0 else 0, noise_count/(np.shape(img)[0]*np.shape(img)[1]))

# """
# COMBINATION OF BLUR AND NOISE

# """

def blur_noise_metric(img):
    #optimized values from study 
    w1 = 1
    w2 = 0.95
    w3 = 0.3
    w4 = 0.75
    blur_mean = Blur_Statistics(img)[0]
    blur_ratio = Blur_Statistics(img)[1]
    noise_mean = Noise_Statistics(img)[0]
    noise_ratio = Noise_Statistics(img)[1]
    #print(f"Blur Mean:{blur_mean}\nBlur Ratio: {blur_ratio}\nNoise Mean: {noise_mean}\nNoise Ratio: {noise_ratio} ")
    return (1-(w1*blur_mean+w2*blur_ratio+w3*noise_mean+w4*noise_ratio))



def test():
    x = []
    y = []
    for file in os.listdir('dicom_test'):
        dt = os.fsdecode(file)
        dt = f"dicom_test/{dt}"
        dicom= pydicom.read_file(dt)
        img = show_ct_dicom(dt)
        EXP = dicom.Exposure
        bnm = blur_noise_metric(img)
        x.append(EXP)
        y.append(bnm)
        
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(x,y, 'mo', x, poly1d_fn(x), '--k')
    x = np.array(x)
    y = np.array(y)
    x = x[:, None]
    model = LinearRegression()
    model.fit(x, y)
    y_predict = model.predict(x)
    corr_matrix = np.corrcoef(y, y_predict)
    corr = corr_matrix[0,1]
    print(corr)

    
    return

test()

# img = show_ct_dicom("t1.dcm")
# plt.figure(figsize = (13,13))
# plt.imshow(img, 'gray')
# print(blur_noise_metric(img))

print(f"Runtime: {time.time()-st}")