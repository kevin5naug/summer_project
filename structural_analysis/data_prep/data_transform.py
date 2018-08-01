import os
import numpy as np

p="/Users/joker/"
if os.path.exists(p):
    os.chdir(p)
input_p="/Users/joker/data/"
output_p="/Users/joker/data_slice/"
L=os.listdir(input_p)
output_index=0
interval=0.1
max_slience=3
#maxx=0
#minx=127
for i,x in enumerate(L):
    input_f=open(input_p+x,"r")
    if len(x)>8:
        continue
    l=input_f.readlines()
    input_f.close()
    for j in range(len(l)):
        l[j]=l[j][:-1].split(" ")
        l[j][0]=float(l[j][0])
        l[j][1]=float(l[j][1])
        l[j][2]=int(float(l[j][2]))
        l[j][4]=int(l[j][4])
        #maxx=max(l[j][2], maxx)
        #minx=min(l[j][2], minx)
    current_interval=0
    current_t=l[0][0]
    end_t=l[-1][1]
    output=[]
    holding=False
    ending=False
    while current_t<=end_t:
        #print(current_t)
        note=np.zeros(86) #0 is hold, 1 is slience, 2-85 is 24-107
        if holding==True:
            note[0]=1
        else:
            holding=True
            if current_t<l[current_interval][0]:
                note[1]=1
            else:
                note[l[current_interval][2]-22]=1
                if ending==True:
                    output[-1][-1]=1
                    ending=False
        temp=[]
        for t in note: temp.append(int(t))
        temp.append(0) #label
        output.append(temp)
        current_t+=interval
        if current_t>l[current_interval][1] and current_t<=end_t:
            holding=False
            if l[current_interval][4]==1: ending=True
            slience_t=l[current_interval+1][0]-l[current_interval][1]
            if slience_t>max_slience:
                current_t+= slience_t-max_slience
            current_interval+=1
    output[-1][-1]=1
    output_f=open(output_p+str(output_index)+".txt", "w")
    for t in output:
        for temp in t:
            output_f.write(str(temp)+" ")
        output_f.write("\n")
    output_index+=1
    output_f.close()
