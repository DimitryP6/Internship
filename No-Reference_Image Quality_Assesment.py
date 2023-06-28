# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:29:19 2023

@author: ddude
"""

import numpy as np
import time
st = time.time()
"""
BLUR
"""
def Difference(image, x, y):
    
    D_h = abs(image[x, y+1]-image[x, y-1])
    D_v = abs(image[x+1, y]-image[x-1, y])
    return (D_h, D_v)

def summation(expression, img):
    sumh = 0
    sumv = 0
    for r in range (2, np.shape(img)[0]-2):
        for c in range (2, np.shape(img)[1]-2):
            sumh += expression(img, r, c)[0]
            sumv += expression(img, r, c)[1]
    return (sumh, sumv)

def summation2(expression, img):
    _sum = 0
    # ignore side two values for candidates 
    for r in range (2, np.shape(img)[0]-2):
        for c in range (2, np.shape(img)[1]-2):
            _sum += expression(img, r, c)
    return (_sum)

def Edge_Candidate(img, x, y):
    D_hm = (summation(Difference, img)[0])/(np.shape(img)[0]*np.shape(img)[1])
    D_vm = (summation(Difference, img)[1])/(np.shape(img)[0]*np.shape(img)[1])
    C_h = Difference(img, x, y)[0] if Difference(img, x, y)[0]>D_hm else 0
    C_v = Difference(img, x, y)[1] if Difference(img, x, y)[1]>D_vm else 0
    return(C_h, C_v)

def Edge_Detection(img, x, y):
    E_h = 1 if Edge_Candidate(img, x, y)[0]>Edge_Candidate(img, x, y-1)[0] and Edge_Candidate(img, x, y)[0]>Edge_Candidate(img, x, y+1)[0] else 0
    E_v = 1 if Edge_Candidate(img, x, y)[1]>Edge_Candidate(img, x-1, y)[1] and Edge_Candidate(img, x, y)[1]>Edge_Candidate(img, x+1, y)[1] else 0
    return (E_h, E_v)

def Bluriness(img, x, y):
    A_h = Difference(img, x, y)[0]/2
    A_v = Difference(img, x, y)[1]/2
    Br_h = abs(img[x, y]-A_h)/A_h if A_h != 0 else 0
    Br_v = abs(img[x, y]-A_v)/A_v if A_h != 0 else 0
    return (max(Br_h, Br_v))


def Blur_Statistics(img):
    #value found suitable in paper
    Th_b = 0.1
    #ignore 2 side pixels for Candidates
    img_sets = np.copy(img)
    img_setc = np.copy(img_sets)
    img_sete = np.copy(img_sets)
    for r in range(2, np.shape(img)[0]-2):
        for c in range(2, np.shape(img)[1]-2):
            #first two statements are blur decision
            img_sets[r, c] = Bluriness(img, r, c) if Bluriness(img, r, c)<Th_b else 0
            img_setc[r, c] = 1 if Bluriness(img, r, c)<Th_b else 0
            img_sete[r, c] = Edge_Detection(img, r, c)[0] if Edge_Detection(img, r, c)[0]==1 else Edge_Detection(img, r, c)[1]
    blur_sum = np.sum(img_sets)
    blur_count = np.sum(img_setc)
    edge_count = np.sum(img_sete)
    #Blur Mean, Blur Ratio
    return(blur_sum/blur_count, blur_count/edge_count)
    

"""
NOISE
"""
# def sigma(start, end, expression):
#     return sum(expression(i,j) for i,j in range (start, end))

# def Average_Filter(img, x, y):
#     noise_sum = 0
#     for r in range(-1, 2):
#         for c in range(-1, 2):
#             noise_sum += img[x+r, y+c]
#     return (noise_sum/9)

def Average_Filter(img, x, y):
    avg = img[(x-1):(x+1), (y-1):(y+1)]
    return np.average(avg)

def Noise_Diff(img, x, y):
    D_h = abs(Average_Filter(img, x, y+1)-Average_Filter(img, x, y-1))
    D_v = abs(Average_Filter(img, x+1, y)-Average_Filter(img, x-1, y)) 
    return(D_h, D_v)

def Noise_Candidate(img, x, y):
    D_hm = (summation(Noise_Diff, img)[0])/(np.shape(img)[0]*np.shape(img)[1])
    D_vm = (summation(Noise_Diff, img)[1])/(np.shape(img)[0]*np.shape(img)[1])
    N_cand = max(Noise_Diff(img, x, y)[0], Noise_Diff(img, x, y)[1]) if Noise_Diff(img, x, y)[0]<=D_hm and Noise_Diff(img, x, y)[1]<=D_vm else 0
    return (N_cand)

def Noise_Decision(img, x, y):
    N_cm = summation2(Noise_Candidate, img)/(np.shape(img)[0]*np.shape(img)[1])
    return Noise_Candidate(img, x, y) if Noise_Candidate(img, x, y)>N_cm else 0

def Noise_Statistics(img):
    #ignore 2 side pixels for Candidates
    img_sets = np.copy(img)
    for r in range(2, np.shape(img)[0]-2):
        for c in range(2, np.shape(img)[1]-2):
            img_sets[r, c] = Noise_Decision(img, r, c)
           
    noise_sum = np.sum(img_sets)
    noise_count = np.count_nonzero(img_sets)
    #Noise Mean, Noise Ratio
    return (noise_sum/noise_count, noise_count/(np.shape(img)[0]*np.shape(img)[1]))
"""
COMBINATION OF BLUR AND NOISE

"""

def blur_noise_metric(img):
    #optimized values from study
    w1 = 1
    w2 = 0.95
    w3 = 0.3
    w4 = 0.75
    print(f"Blur Mean:{Blur_Statistics(img)[0]}\nBlur Ratio: {Blur_Statistics(img)[1]}\nNoise Mean: {Noise_Statistics(img)[0]}\nNoise Ratio: {Noise_Statistics(img)[1]} ")
    return (1-(w1*Blur_Statistics(img)[0]+w2*Blur_Statistics(img)[1]+w3*Noise_Statistics(img)[0]+w4*Noise_Statistics(img)[1]))

test = np.random.randint(256, size=(10, 10))
# print(test)
# print(summation(Difference, test[2:-2, 2:-2]))
print(blur_noise_metric(test))
    


et = time.time()
print(f"Runtime: {et-st}")