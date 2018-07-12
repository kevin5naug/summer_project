import pretty_midi as pyd
import os


min_step = 0.10 # 0.1s
seq_length = 300
file_num = 100
therhold = 40
holding_state = 129

def main():
	path = os.listdir('seqData/RWC/')
	total = 0
	output_stream = []
	token_stream = []
	for d in path:
		total = total + 1
		token_stream = []
		if os.path.isfile('seqData/RWC/' + d):
			while len(token_stream) < seq_length:
				with open('seqData/RWC/' + d,'r') as f:
					for line in f:
						line = line.strip()
						line = line.split()
						q = round(float(line[1]) / min_step)
						if q >= therhold:
							q = therhold
						line.append(q)
						if len(token_stream) > 0 and token_stream[-1] == line[0]:
							token_stream.append(holding_state)
						for i in range(line[2]):
							token_stream.append(line[0])
			token_stream = token_stream[:seq_length]
			output_stream.extend(token_stream)
			print("finish_" + str(total))
	with open('real_data.txt',"w") as q:
		for i in range(len(output_stream)):
			if i % seq_length == 0:
				q.write("\n")
			q.write(str(int(output_stream[i])) + ' ')

	# This is for converting midi to vallina txt file
	# path = os.listdir("metadata/RWC/")
	# total = 0
	# midi_data = pyd.PrettyMIDI('metadata/RWC/RM-P001.SMF_SYNC_MELODY.MID')
	# for i in midi_data.instruments[0].notes:
	# 	print(str(i.pitch) + " " + str(i.start) + " " + str(i.end))
	# for d in path:
	# 	if os.path.isfile('metadata/RWC/' + d):
	# 		total = total + 1
	# 		midi_data = pyd.PrettyMIDI('metadata/RWC/' + d)
	# 		j = 0
	# 		fw = open("seqdata/RWC/train" + str(total) + ".txt","w")
	# 		for i in midi_data.instruments[0].notes:
	# 			if j > 0:
	# 				if round((i.start - j) / min_step) > 0: 
	# 					fw.write("0 " + str(("%.2f" % (i.start - j))) + "\n") 
	# 			fw.write(str(i.pitch) + " " + str(("%.2f" % (i.end - i.start))) + "\n")
	# 			j = i.end
	# 		fw.close()
	# 		print("processed " + str(total))
if __name__ == '__main__':
    main()
