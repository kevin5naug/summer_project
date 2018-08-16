# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pickle as pl
import torch

s = "/Users/Fischer/NYU/MusicStructuralAnalysis_Project/Music_data/"
t = "/Users/Fischer/NYU/MusicStructuralAnalysis_Project/good_data/"
"""
# Select good data
target = "midi.lab.corrected.lab"
target_dir = []
l = os.listdir(s)

for dir in l:
	if os.path.isfile(s + "//" + dir + "//" + target):
		target_dir.append(dir)

for dir in target_dir:
	shutil.copytree(s + dir, t + dir)
"""


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def count_chinese():
    l = os.listdir(t)
    paragraph_info = open(s + l[0] + "/zrc.txt", "r", encoding="utf-8")
    lyrics_info = open(s + l[0] + "/lyric.lab", "r", encoding="utf-16-le")
    content = paragraph_info.readlines()
    ans = 0
    for i in content:
        for j in i:
            if is_chinese(j):
                ans += 1


def file2numpy(file_dir):

    content = target_file.readlines()
    train_y = []
    train_x = []
    time_x = []
    for num, lines in enumerate(content):
        target = []
        for t in lines.split(" "):
            if t != '' and t != '\n':
                target.append(t)
        train_x.append(target[1])
        train_y.append(target[0])
        time_x.append((target[2], target[3]))
    target_file.close()
    return train_x, train_y, time_x


def pitch2numpy(file_dir):
    target_file = open(file_dir, "r", encoding="utf-8", errors="ignore")
    content = target_file.readlines()
    train_y = []
    train_x = []
    max_len=0
    for num, lines in enumerate(content):
        target = []
        for t in lines.split(" "):
            if t != '' and t != '\n':
                target.append(t)
        max_len=max(max_len, int(target[5]))
        train_x.append([float(target[1])-float(target[0]), float(target[2]), float(target[3])])
        train_y.append([int(target[5])])
        #print(target[1], target[0])
    target_file.close()
    return train_x, train_y, max_len

"""
def prepare_data():
    dir = "/Users/Fischer/NYU/MusicStructuralAnalysis_Project/data"
    target_dir = os.listdir(dir)
    train_X = []
    train_Y = []
    time_X = []
    for i in range(len(target_dir)):
        file_dir = dir + "/" + target_dir[i]
        train_x, train_y, time_x = file2numpy(file_dir)
        train_X.append(np.array(train_x))
        train_Y.append(np.array(train_y))
        time_X.append(np.array(time_x))
    train_X, train_Y, time_X = np.array(train_X), np.array(train_Y), np.array(time_X)
    print(train_X.shape, train_Y.shape, time_X.shape)



def prepare_input():
    dir = "/Users/Fischer/NYU/MusicStructuralAnalysis_Project/data"
    target_dir = os.listdir(dir)
    train_X = []
    train_Y = []
    time_X = []
    for i in range(len(target_dir)):
        target_file = open(dir + "/" + target_dir[i], "r", encoding="utf-8", errors="ignore")
        content = target_file.readlines()
        train_y = []
        train_x = []
        time_x = []
        for num, lines in enumerate(content):
            target = []
            for t in lines.split(" "):
                if t != '' and t != '\n':
                    target.append(t)
            train_x.append(target[1])
            train_y.append(target[0])
            time_x.append((target[2], target[3]))

        target_file.close()
        train_X.append(np.array(train_x))
        train_Y.append(np.array(train_y))
        time_X.append(np.array(time_x))
    train_X, train_Y, time_X = np.array(train_X), np.array(train_Y), np.array(time_X)
    print(train_X.shape, train_Y.shape, time_X.shape)
    dic = {"X": train_X, "Y": train_Y, "time": time_X}
    f = open("data.pkl", "wb")
    pl.dump(dic, f)
    f.close()


def find_max_length(train_X):
    max_length = 0
    for i in range(train_X.shape[0]):
        max_length = max(max_length, train_X[i].shape[0])
    return max_length
"""

def pad(vector, pad, dim=0):
    pad_size=list(vector.shape)
    #print(pad_size)
    pad_size[dim]=pad-vector.size(dim)
    #print(pad_size[dim])
    if pad_size[dim]<0:
        print("FATAL ERROR: pad_size=100 not enough!")
    return torch.cat([vector, torch.zeros(*pad_size).type(vector.type())], dim=dim)

def target_factorize(train_X, train_Y, pad_size=120):
    train_X_new=[]
    train_Y_new=[]
    for i, target in enumerate(train_Y):
        total_len=0
        train_X_temp=[]
        train_Y_temp=[]
        one_list=[]
        loc=0
        for label in target:
            if(int(label[0])==1):
                one_list.append(loc)
            loc=loc+1
        prev=0
        print(len(one_list))
        j=0
        while(j<len(one_list)):
            if j==(len(one_list)-1) and prev<one_list[j]:
                x_new=torch.from_numpy(train_X[i][prev:])
                y_new=torch.from_numpy(train_Y[i][prev:].reshape(-1))
                #print(x_new.size())
                print(prev, one_list[j], j, "end")
                total_len+=y_new.size(0)
                train_X_temp.append(pad(x_new, pad_size))
                train_Y_temp.append(pad(y_new, pad_size))
                j+=1
            elif (one_list[j]-prev)<80:
                j+=1
            else:
                x_new=torch.from_numpy(train_X[i][prev:one_list[j-1]])
                y_new=torch.from_numpy(train_Y[i][prev:one_list[j-1]].reshape(-1))
                print(prev, one_list[j-1], j)
                total_len+=y_new.size(0)
                train_X_temp.append(pad(x_new, pad_size))
                train_Y_temp.append(pad(y_new, pad_size))
                prev=one_list[j-1]
        print(total_len)
        train_X_temp=torch.stack(train_X_temp)
        train_Y_temp=torch.stack(train_Y_temp)
        train_X_new.append(train_X_temp)
        train_Y_new.append(train_Y_temp)
    return train_X_new, train_Y_new



def pitch_data():
    pitch_dir = "/Users/joker/data"
    target_dir = os.listdir(pitch_dir)
    train_X = []
    train_Y = []
    max_seq = 0
    for i in range(len(target_dir)):
        if target_dir[i].split(".")[-1] != "txt":
            continue
        #print(target_dir[i].split(".")[-2])
        if int(target_dir[i].split(".")[-2])<=1700:
            continue
        file_dir = pitch_dir + "/" + target_dir[i]
        train_x, train_y, max_len= pitch2numpy(file_dir)
        max_seq=max(max_seq, max_len)
        train_X.append(np.array(train_x))
        train_Y.append(np.array(train_y))
    train_X, train_Y=target_factorize(train_X, train_Y)
    #print(len(train_X), len(train_Y))
    #print(train_X[0])
    dic = {"X": train_X, "Y": train_Y}
    f = open("/Users/joker/pitch_data_validate.pkl", "wb")
    pl.dump(dic, f)
    f.close()
    #print(max_seq)
    return train_X, train_Y


train_X, train_Y= pitch_data()
