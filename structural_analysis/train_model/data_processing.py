# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pickle as pl

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
    for num, lines in enumerate(content):
        target = []
        for t in lines.split(" "):
            if t != '' and t != '\n':
                target.append(t)
        train_x.append([float(target[0]), float(target[1]), float(target[2])])
        train_y.append([int(target[4])])
        #print(target[1], target[0])
    target_file.close()
    return train_x, train_y


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


"""
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
"""


def find_max_length(train_X):
    max_length = 0
    for i in range(train_X.shape[0]):
        max_length = max(max_length, train_X[i].shape[0])
    return max_length



def pitch_data():
    pitch_dir = "/Users/joker/data"
    target_dir = os.listdir(pitch_dir)
    train_X = []
    train_Y = []
    for i in range(len(target_dir)):
        if target_dir[i].split(".")[-1] != "txt":
            continue
        file_dir = pitch_dir + "/" + target_dir[i]
        train_x, train_y= pitch2numpy(file_dir)
        train_X.append(np.array(train_x))
        train_Y.append(np.array(train_y))
    train_X, train_Y= np.array(train_X), np.array(train_Y)
    print(train_X.shape, train_Y.shape)
    dic = {"X": train_X, "Y": train_Y}
    f = open("pitch_data.pkl", "wb")
    pl.dump(dic, f)
    f.close()
    return train_X, train_Y


train_X, train_Y= pitch_data()
max_length = find_max_length(train_X)
print(max_length)
