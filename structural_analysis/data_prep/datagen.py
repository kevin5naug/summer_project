import os
def generate_data(path, target_file):
    os.chdir(path)
    for subdir_name in os.listdir(path):
        subdir=path+subdir_name
        if os.path.isdir(subdir):
            os.chdir(subdir)
            if contain_target_file(os.listdir(subdir), target_file):
                path_input=subdir+"/"
                path_output="/Users/joker/data/"
                parse_file(path_input, path_output)

def contain_target_file(file_list, target):
    for fname in file_list:
        if os.path.isfile(fname):
            if fname==target:
                return True
    return False

def parse_file(path, output_path):
    p1=path+"zrc.txt"
    f1=open(p1, "r")
    content1=f1.readlines()
    p2=path+"lyric.lab"
    f2=open(p2, "r", encoding="utf-16-le")
    content2=f2.readlines()
    
    #First Pass
    ds1=DataSolverPass1()
    ds1.digest(content1)
    data_first_pass=[]
    for line in content2:
        begin_t, end_t, target_char=extract_info(line)
        target_index=ds1.match(target_char)
        data_first_pass.append([target_index, target_char, begin_t, end_t])
    
    #Second Pass
    data_second_pass=[]
    ds2=DataSolverPass2()
    ds2.digest(data_first_pass)
    normalize_t1=-1
    for index in range(len(data_first_pass)):
        current_point=data_first_pass[index]
        if normalize_t1==-1:
            normalize_t1=float(current_point[2])
        start, end=ds2.match(current_point[0])
        label=determine_label(start, end, index)
        data_second_pass.append([label, current_point[1], float(current_point[2]), float(current_point[3])])
    
    #Third Pass
    p3=path+"midi.lab.corrected.lab"
    f3=open(p3, "r")
    content3=f3.readlines()
    data_third_pass=[]
    end_t_max=data_second_pass[-1][3]
    ds3=DataSolverPass3()
    normalize_t2=ds3.digest(content3, end_t_max)
    for index in range(len(data_second_pass)):
        current_point=data_second_pass[index]
        label, char, begin_t, end_t=current_point[0], current_point[1], current_point[2], current_point[3]
        ds3.match(begin_t, end_t, char, label)
    ds3.mark_the_rest()
    for index in range(len(ds3.note_sequence)):
        begin_t, end_t, pitch=ds3.note_sequence[index]
        label, char=ds3.label_sequence[index], ds3.char_sequence[index]
        data_third_pass.append([begin_t, end_t, pitch, char, label])
    
    #Output data file
    output_path+=path[-5:-1]
    output_path+=".txt"
    thefile=open(output_path,'w')
    for point in data_third_pass:
        thefile.write("%10s %10s %10s %10s %10s\n" % (point[0], point[1], point[2], point[3], point[4]))
    thefile.close()

def extract_info(line):
    items=(line.strip()).split("\t")
    if '\ufeff' in items[0]:
        items[0]=items[0][len("\ufeff"):]
    return items

def determine_label(start, end, index):
    if ((start==end) and (start==index)):
        return 1 #single
    elif ((start<end) and (start<=index) and (index<=end)):
        if start==index:
            return 2 #start
        elif end==index:
            return 4 #end
        else:
            return 3 #in the middle of a sentence
    else:
        print(start, end, index)
        print("FATAL: LABEL ERROR -1")
        return -1

class DataSolverPass1():
    def __init__(self):
        self.search_record={}
        self.item_positions={}
    def digest(self, content):
        for index in range(len(content)):
            current_line=content[index]
            item=""
            start_flag=False
            for char in current_line:
                if char==">":
                    start_flag=True
                elif ((char=="<") or (char=="\n")):
                    start_flag=False
                    if item not in self.item_positions:
                        self.item_positions[item]=[]
                    self.item_positions[item].append(index)
                    item=""
                else:
                    if start_flag:
                        item=item+char

    def match(self, target):
        if target not in self.search_record:
            self.search_record[target]=0
        else:
            self.search_record[target]+=1
        index=self.search_record[target]
        if target in self.item_positions:
            return self.item_positions[target][index]
        else:
            return -1 #no match found

class DataSolverPass2():
    def __init__(self):
        self.start_positions={}
        self.end_positions={}
    def digest(self, content):
        sentence_index=-1
        for index in range(len(content)):
            current_point=content[index]
            sentence_index=current_point[0]
            if sentence_index not in self.start_positions:
                self.start_positions[sentence_index]=index
                self.end_positions[sentence_index-1]=index-1
        # the end position of the last sentence is a special case
        if sentence_index not in self.end_positions:
            self.end_positions[sentence_index]=len(content)-1
    def match(self, target):
        if ((target in self.start_positions) and (target in self.end_positions)):
            return (self.start_positions[target], self.end_positions[target])
        else:
            return (-1,-1)

class DataSolverPass3():
    def __init__(self):
        self.note_sequence=[]
        self.label_sequence=[]
        self.char_sequence=[]
        self.end_t_max=-1
        self.normalize_t=-1
        self.pointer=0
    def digest(self, line_sequence, end_t_max):
        self.end_t_max=end_t_max
        for line in line_sequence:
            begin_t, end_t, pitch=extract_info(line)
            begin_t, end_t, pitch=float(begin_t), float(end_t), int(float(pitch))
            if self.normalize_t==-1:
                self.normalize_t=begin_t
            if end_t<=self.end_t_max:
                self.note_sequence.append([begin_t, end_t, pitch])
                self.label_sequence.append(-1)
                self.char_sequence.append("")
        return self.normalize_t
    def match(self, target_begin, target_end, target_char, target_label):
        self.find_earliest_note(target_begin, target_end, target_char, target_label)
    def find_earliest_note(self, target_begin, target_end, target_char, target_label):
        start_range=self.pointer
        for index in range(start_range, len(self.note_sequence)):
            current_note=self.note_sequence[index]
            current_note_begin, current_note_end=current_note[0], current_note[1]
            if(getOverlap(current_note_begin, current_note_end, target_begin, target_end)>0):
                if ((self.label_sequence[index]==-1) and (self.char_sequence[index]=="")):
                    self.label_sequence[index]=target_label
                    self.char_sequence[index]=target_char
                    self.pointer=index+1
                    return
                else:
                    print("ERROR: this note has already been assigned a label/char")
                    return
    def mark_the_rest(self):
        in_sentence=False
        for index in range(len(self.note_sequence)):
            current_note=self.note_sequence[index]
            label=self.label_sequence[index]
            if label==-1:
                if in_sentence:
                    self.label_sequence[index]=3
                    self.char_sequence[index]="~"
                else:
                    self.label_sequence[index]=0
                    self.char_sequence[index]="~"
            elif label==2:
                in_sentence=True
            elif label==4:
                in_sentence=False
            elif label in [1,3]:
                pass
            else:
                print("ERROR: unkowned label type encountered")
        return
        


        
def getOverlap(current_begin, current_end, target_begin, target_end):
    return max(0, min(current_end, target_end)-max(current_begin, target_begin))

p="/Users/joker/Downloads/data/"
target="midi.lab.corrected.lab"
generate_data(p, target)
