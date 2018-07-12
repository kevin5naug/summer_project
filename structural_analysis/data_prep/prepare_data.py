import os
p="/Users/joker/Downloads/data/"
target="midi.lab.corrected.lab"
os.chdir(p)
count=0
for dname in os.listdir(p):
    s=p+dname
    if os.path.isdir(s):
        os.chdir(s)
        for fname in os.listdir(s):
            if os.path.isfile(fname):
                if fname==target:
                    count+=1
    else:
        continue
print(count)

