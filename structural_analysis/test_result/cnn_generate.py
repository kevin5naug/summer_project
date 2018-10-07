from midiutil.MidiFile import MIDIFile 
import pickle
import torch
import numpy as np

with open("/Users/joker/pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]

with open("/Users/joker/33557.pkl", "rb") as f:
	prediction = pickle.load(f)

print(len(prediction))

def cnn_midi_generator(X, Y, Z, n):
	track = 0
	channel = 0
	volume = 100
	MyMIDI = MIDIFile(3)
	label_track = 1
	prediction_track = 2
	time = 0
	tempo = 60
	MyMIDI.addTempo(track,time, tempo)
	for i, item in enumerate(X[n]):
		time = float(item[0])
		duration = float(item[1]) - float(item[0])
		#time = i
		#duration = 1
		pitch = int(item[2])
		flag = int(Y[n][i])
		MyMIDI.addNote(track,channel,pitch,time,duration,volume)
		if(flag == 4):
			MyMIDI.addNote(label_track,channel,90,time,duration,volume)
		flag = int(Z[n - 1300][i])
		if(flag == 1):
			MyMIDI.addNote(prediction_track,channel,30,time,duration,volume)

	midifile = open("/Users/joker/33557_p/CNN_sample{}.mid".format(int(n)), "wb")
	MyMIDI.writeFile(midifile)
	midifile.close()

for i in range(1300, 1373):
	cnn_midi_generator(train_X, train_Y, prediction, i)
