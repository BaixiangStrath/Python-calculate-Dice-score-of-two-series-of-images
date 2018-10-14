# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:34:36 2018

@author: Administrator
"""

import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from sklearn.metrics import confusion_matrix

label_Path = "yourPath/testLabel"
test_Path = "YourPath/testPredict"


def diceScoreCal(test_Path,label_Path,num_image=60,target_size = (512,512),as_gray = True,data_type = np.bool_):
    dice_score = []
    img_size = target_size[0]*target_size[1]
    for i in range(num_image):
         img_test = io.imread(os.path.join(test_Path,"%d_predict.png"%i),as_gray = as_gray)
         img_test = trans.resize(img_test,target_size)
         img_test = img_test.astype(data_type)
         img_label = io.imread(os.path.join(label_Path,"%d.png"%i),as_gray = as_gray)
         img_label = trans.resize(img_label,target_size)
         img_label = img_label.astype(data_type)
         cf_mtrx = confusion_matrix(np.squeeze(np.reshape(img_label,(1,img_size))),np.squeeze(np.reshape(img_test,(1,img_size))),labels = [1,0])
         TP = cf_mtrx[0][0]
         FP = cf_mtrx[1][0]
         FN = cf_mtrx[0][1]
         TN = cf_mtrx[1][1]    
         Dice_sc = 2*TP/(2*TP+FP+FN)
         dice_score.append(Dice_sc)
    return dice_score




ds = diceScoreCal(test_Path,label_Path)

ds_mean = np.average(ds)
