import os
import annoy
import pandas as pd
import numpy as np
from annoy import AnnoyIndex



loc = "E:\\DataBase\\Positive\\Hog\\"
save_loc = "E:\\DataBase\\Ann\\"
loc2 = "E:\\DataBase\\Positive\\Gist\\"
locn = os.listdir(loc)
print 'loop here',locn
for i in locn:
	print 'annoy'
	counter=0
	f = 360+512
	t = AnnoyIndex(f, metric = 'angular')	
	feat = loc + i
	feat2 = loc2+i
	annl = i.split('.')[0]
	feat1 = pd.read_csv(feat,sep=',',header=None)
	feat1=np.asarray(feat1)
	feat3 = pd.read_csv(feat2,sep=',',header=None)
	feat3 = np.asarray(feat3)
	hogx,hogy = feat3.shape
	feat1= feat1[:hogx,:]
	
	print 'shapes are',feat1.shape,feat3.shape
	feat1 = np.concatenate((feat1,feat3),axis=1)
	print feat1.shape
	row,col = feat1.shape
	feat1 = feat1[:,:col-2]
	for count2 in range(row):
		count = feat1[count2,:]
		
		t.add_item(counter, count.astype(np.int32))
		counter=counter+1
	t.build(100)
	t.save(save_loc+annl+'.ann')
	t.unload()
	del t
	print "Done with"+str(i)
