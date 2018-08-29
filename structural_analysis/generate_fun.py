from midiutil.MidiFile import MIDIFile
import mido
import numpy as np
import torch
import pickle as pl
from mido import MidiFile
note = []
time = []
mid = MidiFile("/Users/joker/Croatian.mid")
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    s = 0
    if i == 3:
        for msg in track:
            print(msg.bytes())
            note.append(msg.bytes()[1])
            time.append(msg.time)
"""
midifile = open("test.mid", "wb")
MyMIDI.writeFile(midifile)
midifile.close()"""

X = np.zeros((len(note), 3))
for i, n in enumerate(note):
    X[i,2] = n
    X[i,1] = 0
    X[i,0] = (time[i] + 1) / 64.0
X = torch.from_numpy(X)
f=open("/Users/joker/test.pkl", "wb")
pl.dump(X, f)
f.close()
print(X.size())

with open("/Users/joker/Coding/summer_project/structural_analysis/prediction.pkl", "rb") as f:
        lstm_prediction = pl.load(f)

def crf_midi_generator(X, Z, n, m, silence, note_length):
        track = 0
        channel = 0
        volume = 100
        MyMIDI = MIDIFile(3)
        label_track = 0
        prediction_track = 1
        time = 0
        tempo = 60
        MyMIDI.addTempo(track,time, tempo)
        print(Z["out"][m])
        print(X.shape)
        count = 0
        total = 0
        current_time=0
        for i, item in enumerate(X):
                if(i>=len(Z['out'][m])):
                        continue
                duration = float(item[0])
                time=current_time
                current_time+=duration
                #time = i
                #duration = 1
                pitch = int(item[2])
                print(pitch)
                MyMIDI.addNote(track,channel,pitch,time,duration,volume)
                flag = int(Z["out"][m][i])
                if(flag == 1):
                        MyMIDI.addNote(prediction_track,channel,30,time,duration,volume)
                        #print(X[n][i-1][1]-X[n][i-1][0], X[n][i][0]-X[n][i-1][1])            
                        if i>0 and (X[i-1][1]-X[i-1][0])<note_length and (X[i][0]-X[i-1][1])<silence:
                                count+=1    
        path = "/Users/joker/sample" +str(n)+"_"+str(m)+".mid"
        midifile = open(path, "wb")
        MyMIDI.writeFile(midifile)
        midifile.close()
        return count, total

count, total=crf_midi_generator(X, lstm_prediction, 0, 0, 0.01, 0.3)
