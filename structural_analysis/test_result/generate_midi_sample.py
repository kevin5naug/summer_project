from midiutil.MidiFile import MIDIFile 
import pickle
import torch
import numpy as np

with open("/Users/joker/pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]

with open("/Users/joker/cnncrf_prediction.pkl", "rb") as f:
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


def crf_midi_generator(X, Y, Z, n, m, silence, note_length):
	track = 0
	channel = 0
	volume = 100
	MyMIDI = MIDIFile(3)
	label_track = 1
	prediction_track = 2
	time = 0
	tempo = 60
	MyMIDI.addTempo(track,time, tempo)
	print(Z["out"][m].shape[0])
	print(X[n].shape)
	count = 0
	total = 0
	for i, item in enumerate(X[n]):
		if(i>=Z['out'][m].shape[0]):
			continue
		time = float(item[0])
		duration = float(item[1]) - float(item[0])
		#time = i
		#duration = 1
		pitch = int(item[2])
		flag = int(Y[n][i])
		MyMIDI.addNote(track,channel,pitch,time,duration,volume)
		if(flag == 2):
			MyMIDI.addNote(label_track,channel,90,time,duration,volume)
			if i>0 and (X[n][i-1][1]-X[n][i-1][0])<note_length and (X[n][i][0]-X[n][i-1][1])<silence:
				total+=1            
		flag = int(Z["out"][m][i])
		if(flag == 1):
			MyMIDI.addNote(prediction_track,channel,30,time,duration,volume)
			#print(X[n][i-1][1]-X[n][i-1][0], X[n][i][0]-X[n][i-1][1])            
			if i>0 and (X[n][i-1][1]-X[n][i-1][0])<note_length and (X[n][i][0]-X[n][i-1][1])<silence:
				count+=1    
	path = "/Users/joker/binary_cnncrf/sample" +str(n)+"_"+str(m)+".mid"
	midifile = open(path, "wb")
	MyMIDI.writeFile(midifile)
	midifile.close()
	return count, total    

total_val=0
count_val=0
for i in range(150):
    count, total=crf_midi_generator(train_X, train_Y, lstm_prediction, i, i, 0.01, 0.3)
    total_val+=total
    count_val+=count

print("model diagnose: ", count_val) 
print("total critical targets num: ", total_val)
print("ratio: ", float(count_val)/total_val)

