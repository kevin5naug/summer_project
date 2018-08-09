import os
import numpy as np
import pickle as pl
import random

PATTERN_LEN=5

def generate_sentence(start_t):
    sequence_length=random.randint(7,11)
    pattern=random.randint(0,1)
    train_X=[]
    train_Y=[]
    end_note=0
    lasting=0
    end_t=0
    for i in range(sequence_length):
        #Suppose a sentence ends when five notes monotonically increase/decrease in a series
        if ((i+PATTERN_LEN-1)<(sequence_length-1)):
            lasting=random.randint(2,7) #this note lasts 1-5 seconds
            train_X.append([lasting, random.randint(30,90)])
            train_Y.append(i+1)
            start_t+=lasting
        elif((i+PATTERN_LEN-1)==(sequence_length-1)):
            end_note=random.randint(30,40)
            lasting=random.randint(2,7) #this note lasts 1-5 seconds
            train_X.append([lasting, end_note])
            train_Y.append(i+1)
            start_t+=lasting
            if pattern>0:
                end_note+=1
            else:
                end_note-=1
        elif(i<(sequence_length-1)):
            lasting=random.randint(2,7) #this note lasts 1-5 seconds
            train_X.append([lasting, end_note])
            train_Y.append(i+1)
            start_t+=lasting
            if pattern>0:
                end_note+=1
            else:
                end_note-=1
        elif(i==(sequence_length-1)):
            lasting=random.randint(2,7) #this note lasts 1-5 seconds
            train_X.append([lasting, end_note])
            train_Y.append(i+1)
            end_t=start_t+lasting
        else:
            print("FATAL ERROR")
    return train_X, train_Y, end_t

def generate_data(min_len):
    count=0
    train_X=[]
    train_Y=[]
    start_t=0
    end_t=0
    while(count<min_len):
        train_X0, train_Y0, end_t=generate_sentence(start_t)
        count+=len(train_X0)
        train_X+=train_X0
        train_Y+=train_Y0
        start_t=end_t
    train_X=np.array(train_X)
    train_Y=np.array(train_Y)
    print(train_X, train_Y)
    dic={"X": train_X, "Y":train_Y}
    f=open("/Users/joker/toy_data.pkl", "wb")
    pl.dump(dic, f)
    f.close()

generate_data(1000000)


