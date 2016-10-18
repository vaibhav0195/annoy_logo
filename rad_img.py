import dummy5
import cv2
import pandas as pd
import numpy as np

print 'inmain'
img1 = cv2.imread("E:\\img1.png")
print 'loaded image'

data_gist = pd.read_csv("E:\\DataBase\\Positive\\Gist\\Air india.csv",sep=',',header=None)
data_hog = pd.read_csv("E:\\DataBase\\Positive\\Hog\\Air india.csv",sep=',',header=None)
data_gist = np.asarray(data_gist)
data_hog = np.asarray(data_hog)
temp1 = data_gist[1,:512]
temp2 = data_hog[1,:360]
temp3 = np.concatenate((temp1,temp2))
dummy5.combine_labels(['Air india','Aston martin'],temp3)
#temp_list = dummy5.image_calc(img1)
#print temp_list