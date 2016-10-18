import Gist_feat_last
import HOG_feat2
import os
import numpy as np
from sklearn.externals import joblib  #save the data
import cv2
import json
from annoy import AnnoyIndex

'''from pymongo import MongoClient
with open('Properties.json', 'r') as fp:
    data = json.load(fp)

dir_gist = data["ClassifierGist"]
dir_hog = data["ClassifierHog"]
m_a = data["MongoUrl"]

c = MongoClient(m_a)   #taking instance of mongo client
mer4 = data["ImageDatabase"]
db = c[mer4] 
db_classig = db.ClasiGabor
db_classih = db.ClassiHog'''
list1 = {} #dictionary to hold data
list2 = {}
list3 = {}
files_name = []
ann_add= "E:\\DataBase\\Ann"
dir_gist = "E:\\DataBase\\Classifier\\Gist\\"
dir_hog = "E:\\DataBase\\Classifier\\Hog\\"
#just return the name of company of classiffier
def remove_num(list_temp):
    for i in range(len(list_temp)):
        temp = list_temp[i]
        temp1 = temp.split('_')
        temp = temp1[0]
        list_temp[i] = temp
        
    return list_temp

def ret_key(d23,top):
    s = []
    for i in d23:
        if d23[i] >=top :
            s.append(i)
    return s

def make_occurence_table(keys,comb):
    occ_table = {}
    for i in keys:
        occ_table[i] = 0
    for i in comb: 
        if occ_table.has_key(i):
            occ_table[i] = occ_table[i]+1
    return occ_table

def final_labels(names_list,f_type):
    ann_considered = []
    ret_list = []
    distances_considered = {}
    distances = []
    if (f_type == "Gist"):
        f = 512
        t = AnnoyIndex(f, metric = 'angular')
    elif (f_type == "Hog"):
        f=360
        t = AnnoyIndex(f,metric = 'angular')
    ann_add_temp = ann_add+"\\"+f_type
    ann_names = os.listdir(ann_add_temp)
    for ann_n in ann_names:
        ann_considered.append(ann_n.split('.')[0])
    for ann_names3 in ann_considered:
        if ann_names3 in names_list:
            ret_list.append(ann_names3)
    for rets in ret_list:
        ann_load = ann_add_temp+"\\"+rets+".ann"
        t.load(ann_load)
        temp_dist = t.get_nns_by_item(a1.astype(np.int32), 1, -1, True)
        distances_considered[rets] = temp_dist
        distances.append(temp_dist)
    return distances_considered,distances

def combine_labels(names_list,annoyfeature):
    ann_considered = []
    ret_list = []
    distances_considered = {}
    distances = []
    print 'in function',annoyfeature.shape
    f= 360+512
    
    print 'annoy made'
    ann_names = os.listdir(ann_add)
    for ann_n in ann_names:
        ann_considered.append(ann_n.split('.')[0])
    print 'considered'
    for ann_names3 in ann_considered:
        if ann_names3 in names_list:
            ret_list.append(ann_names3)
    print 'list is',ret_list
    for rets in ret_list:
        ann_load = ann_add+"\\"+rets+".ann"
        print 'loading is',ann_load
        t2 = AnnoyIndex(f,metric = 'angular')
        t2.load(ann_load)
        temp_dist = t2.get_nns_by_item(annoyfeature.astype(np.int32), 1,search_k=-1, include_distances=True)
        #print t.get_nns_by_item(annoyfeature, 1,search_k=-1, include_distances=True)
        print 'annot dist',temp_dist
        t2.unload()
        distances_considered[rets] = temp_dist
        distances.append(temp_dist)
    print 'list and dic is'
    print distances_considered,distances
    return distances_considered,distances
        


def Label_classify(feature,files1):
    dir2 = dir_gist #directory where the classifier are
    for subdir2,newdir1,files3 in os.walk(dir2):
        list1[files1]=[]
        files_name.append(files1)
        for files4 in files3:
            machine_path = dir2+files4
            clf = joblib.load(machine_path) #load the classifier
            predict = clf.predict(feature) #predict the class
            predict = np.asarray(predict)
            if predict.all()==1:  #if class is one then add it
                
                list1[files1].append(files4)
'''    res = db_classig.find()
    res = json.loads(dumps(res))
    for r in res:
        count = 0
        for j in r:
            if j != "name":
                clf = r[j]
                predict = clf.predict(feature)
                predict = np.asarray(predict)
                if predict.all()==1:
                    count=count+1
                    temp = r["name"]
                    list1[files1].append(temp+'_'+str(count))'''


                
def Label_classify2(feature,files1):
    dir2 = dir_hog #directory where the classifier are
    for subdir2,newdir1,files3 in os.walk(dir2):
        list2[files1]=[]

        for files4 in files3:
            machine_path = dir2+'\\'+files4
            clf = joblib.load(machine_path) #load the classifier
            predict = clf.predict(feature) #predict the class
            predict = np.asarray(predict)
            #print predict
            if predict.all()==1:  #if class is one then add it
                #print 'Prediction is:',files4
                list2[files1].append(files4)
'''   res = db_classih.find()
    res = json.loads(dumps(res))
    for r in res:
        count = 0
        for j in r:
            if j != "name":
                clf = r[j]
                predict = clf.predict(feature)
                predict = np.asarray(predict)
                if predict.all()==1:
                    count=count+1
                    temp = r["name"]
                    list1[files1].append(temp+'_'+str(count))'''

def image_calc(img):
	
    '''s1 = str2
    s4 = s1.split('path')[1]
    s4 = s4[1:]
    s4 = s4.split('>')[0]
    str2 = s4.replace('//', '\\')
    print 'image s',str2'''
    try:

        #img = cv2.imdecode(str2
        #img = cv2.imread(str2)
        global list1
        global list2
        global files_name
        correct_fea = Gist_feat_last.singleImage2(img)
        feat = HOG_feat2.hog_call(img)
        annoyfeature = np.concatenate((correct_fea,feat))
        #print 'hog',feat.shape
        print 'Feature extracted'
        Label_classify2(feat,'batman')
        Label_classify(correct_fea,'batman')
        #print 'list',list1
        #print '2nd list',list2
        temp_list1 = []
        temp_list2 = []
        last_list=[]
        ret_list1 = []
        ret_list2 = []
        print list1
        print list2
        for file12 in files_name:
            print 'in loop'
            names = list1[file12]
            names2 = list2[file12]
            print 'got  the list'
            list1 = remove_num(names)
            list2 = remove_num(names2)
            list1_set = set(list1)
            list2_set = set(list2)
            print 'occ table made'
            o1 = make_occurence_table(list1_set,list1)
            o2 = make_occurence_table(list2_set,list2)
            temp_list2 = ret_key(o1,2)
            temp_list1 = ret_key(o2,2)

            last_list = list(set(list1).intersection(set(list2)))
        #lab1,distances = final_labels(temp_list1,'Hog')
        #lab2,distances2 = final_labels(temp_list2,'Gist')
        #hog_min = distances.min()
        #gist_min = distances2.min()
        #print 'lst',last_list,annoyfeature
        print 'b4 annoy'
        lab1,distances = combine_labels(last_list,annoyfeature)
        print "after annoy"
        mindist = distances.min()
        for i in lab1:
            if lab1[i] == hog_min:
                ret_list2.append(i)
        #for i in lab2:
        #    if lab2[i] == gist_min:
        #        ret_list1.append(i)
        #print 'last',list3
        #last_list = list(set(ret_list1).intersection(ret_list2))
        return ret_list2

    except Exception,e:
        return 'Image not found',e 
	
