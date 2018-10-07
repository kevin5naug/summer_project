from midiutil import MIDIFile

degrees  = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note number
track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = 60   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, time, tempo)

f=open("/Users/joker/Downloads/data/0003/midi.lab.corrected.lab", "r")
content=f.readlines()
for line in content:
    line=line.strip()
    [begin, end, pitch]=line.split()
    MyMIDI.addNote(track, channel, int(float(pitch)), float(begin), float(end)-float(begin), volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
