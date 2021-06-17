from openpyxl import load_workbook
import random
import  numpy as np
from PIL import Image
import os
import csv
def extractone(sector):
    # with open(r"D:\QQ\1392776996\FileRecv\21days_data_with_cplx\21days_data_with_cplx\SectorFactor1002\%s" %sector,'r') as csvfile:
    with open(sector,'r') as csvfile:
        reader = csv.reader(csvfile)
        # print(reader)
        i=0
        rows=[]
        shouldextractfeature= [0,1,3,4,6,7,8,9,10,13,14,15,18,20,22,24,43]
        for row in reader:
            tmprow=[]
            for i in shouldextractfeature:
                tmprow.append(row[i])
            rows.append(tmprow)
        csvfile.close()
    # temp = np.array(rows)
    return rows
# temp.
#  print(f'x shape: {x.shape}, y shape: {y.shape}')
path=r"D:\QQ\1392776996\FileRecv\21days_data_with_cplx\21days_data_with_cplx"
dirs = os.listdir(path)
alldata=[]
for dir in dirs:
    shanqu=os.path.join(path,dir)
    shanqudirs=os.listdir(shanqu)
    singlgday=[]
    for shanqudir in shanqudirs:
        sectordatapath = os.path.join(shanqu, shanqudir)
        # print(sectordatapath)
        tmp = extractone(sectordatapath)
        singlgday.append(tmp)
        # temp = np.array(tmp)
        # print(temp.shape,len(alldata))
        # alldata.append(tmp)
        # print(len(alldata))
    alldata.append(singlgday)
    temp = np.array(alldata)
    temp1 = np.array(singlgday)
    print(temp.shape,temp1.shape)
alltosave = np.array(alldata)
print(alltosave.shape)
np.save("alldayfeature.npy", alltosave)
print("save .npy done")
        # print(tmp.shape)
    # print(shanqu)
# tmp=extractone("sector1.csv")
# print(tmp.shape)
