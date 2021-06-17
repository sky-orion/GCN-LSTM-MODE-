from openpyxl import load_workbook
import random
import numpy as np
from PIL import Image
import os
import csv
# arr=np.load("alldayfeature.npy").transpose((0,2,1,3))
# arr=np.zeros((21, 126, 1440, 17)).transpose((0,2,1,3))
arr=np.load("all_day_feature_single_daytime.npy",mmap_mode = 'r')
# arr.dtype="float"
arr=arr.astype(np.float)
print(arr.dtype,arr.shape,arr[0,0,:])
# print(arr.shape)
# global tmp
# for i in range(arr.shape[0]):
#     # tmp.dtype = 'float'
#     if i==0:
#         tmp=arr[i,:,:,:]
#     else:
#         tmp=np.concatenate((tmp,arr[i,:,:,:]),axis=0)
# print(tmp.shape)
# np.save("all_day_feature_single_daytime.npy", tmp)
# print("save .npy done")