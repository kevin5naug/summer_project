from midiutil.MidiFile import MIDIFile 
import pickle
import torch
import numpy as np

with open("/Users/joker/pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]

with open("/Users/joker/prediction.pkl", "rb") as f:
	lstm_prediction = pickle.load(f)


def midi_generator(X, Y, Z, n):
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
		flag = int(Z[n][i])
		if(flag == 1):
			MyMIDI.addNote(prediction_track,channel,30,time,duration,volume)

	midifile = open("/Users/joker/lstmcrf_v0/music_prediction/sample{}.mid".format(int(n)), "wb")
	MyMIDI.writeFile(midifile)
	midifile.close()


def crf_midi_generator(X, Y, Z, n):
	track = 0
	channel = 0
	volume = 100
	MyMIDI = MIDIFile(3)
	label_track = 1
	prediction_track = 2
	time = 0
	tempo = 60
	MyMIDI.addTempo(track,time, tempo)
	print(Z["out"][2].shape)
	print(X[n].shape)
	for i, item in enumerate(X[n]):
		time = float(item[0])
		duration = float(item[1]) - float(item[0])
		#time = i
		#duration = 1
		pitch = int(item[2])
		flag = int(Y[n][i])
		MyMIDI.addNote(track,channel,pitch,time,duration,volume)
		if(flag == 2):
			MyMIDI.addNote(label_track,channel,90,time,duration,volume)
		flag = int(Z["out"][0][i])
		if(flag == 1):
			MyMIDI.addNote(prediction_track,channel,30,time,duration,volume)

	midifile = open("sample.mid", "wb")
	MyMIDI.writeFile(midifile)
	midifile.close()

crf_midi_generator(train_X, train_Y, lstm_prediction, 1285)

