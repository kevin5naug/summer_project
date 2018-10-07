def parse_file(path, output_path):
    p1=path+"zrc.txt"
    f1=open(p1, "r")
    content1=f1.readlines()
    p2=path+"lyric.lab"
    f2=open(p2, "r", encoding="utf-16-le")
    content2=f2.readlines()
    ds1=DataSolverPass1()
    ds1.digest(content1)
    data_first_pass=[]
    for line in content2:
        begin_t, end_t, target_char=extract_info(line)
        target_index=ds1.match(target_char)
        data_first_pass.append([target_index, target_char, begin_t, end_t])
    data_second_pass=[]
    ds2=DataSolverPass2()
    ds2.digest(data_first_pass)
    for index in range(len(data_first_pass)):
        current_point=data_first_pass[index]
        start, end=ds2.match(current_point[0])
        label=determine_label(start, end, index)
        data_second_pass.append([label, current_point[1], current_point[2], current_point[3]])
    output_path+=path[-5:-1]
    output_path+=".txt"
    thefile=open(output_path,'w')
    for point in data_second_pass:
        thefile.write("%10s %10s %10s %10s \n" % (point[0], point[1], point[2], point[3]))
    thefile.close()

def extract_info(line):
    items=(line.strip()).split("\t")
    if '\ufeff' in items[0]:
        items[0]=items[0][len("\ufeff"):]
    return items

def determine_label(start, end, index):
    if ((start==end) and (start==index)):
        return 0 #single
    elif ((start<end) and (start<=index) and (index<=end)):
        if start==index:
            return 1 #start
        elif end==index:
            return 3 #end
        else:
            return 2 #in the middle of a sentence
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

#p="/Users/joker/Downloads/data/0003/"
#parse_file(p, "/Users/joker/data/")
