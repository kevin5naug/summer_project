import os
p="/Users/joker/Downloads/data/"
target="midi.lab.corrected.lab"
os.chdir(p)
for dname in os.listdir(p):
    if os.path.isdir(dname):
        oldname=os.path.basename(dname)
        newname=dname[0:4]
        os.rename(p+oldname, p+newname)
print("success")
