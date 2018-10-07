import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle
from copy import deepcopy
import random

device = torch.device(3 if torch.cuda.is_available() else "cpu")
#CRF Parameters Pre-set
SEQ_LEN=120 #Training set sequence length
BATCH_SIZE=256 #Training batch size
VAL_SEQ_LEN=880
VAL_BATCH_SIZE=1
CLIP = 5
input_dim=3
output_size=60
START_TAG=output_size-2
STOP_TAG=output_size-1
hidden_dim=512

def training_set_factorize(train_X, train_Y, pad_size=120, augment_data=True):
    train_X_new=[]
    train_Y_new=[]
    for i, target in enumerate(train_Y):
        total_len=0
        one_list=[]
        loc=0
        for label in target:
            if(int(label[0])==1):
                one_list.append(loc)
            loc=loc+1
        prev=0
        #print(one_list)
        j=0
        while(j<len(one_list)):
            if j==(len(one_list)-1) and prev<one_list[j]:
                x_new=torch.from_numpy(train_X[i][prev:])
                y_new=torch.from_numpy(train_Y[i][prev:].reshape(-1))
                #print(x_new.size())
                #print(prev, one_list[j], j, "end")
                total_len+=y_new.size(0)
                train_X_new.append(x_new)
                train_Y_new.append(y_new)
                j+=1
            elif (one_list[j]-prev)<80:
                j+=1
            else:
                x_new=torch.from_numpy(train_X[i][prev:one_list[j-1]])
                y_new=torch.from_numpy(train_Y[i][prev:one_list[j-1]].reshape(-1))
                #print(prev, one_list[j-1], j)
                total_len+=y_new.size(0)
                train_X_new.append(x_new)
                train_Y_new.append(y_new)
                prev=one_list[j-1]
    train_X_augment=[]
    train_Y_augment=[]
    for i, target in enumerate(train_Y_new):
        train_X_augment.append(pad(train_X_new[i], pad_size))
        train_Y_augment.append(pad(train_Y_new[i], pad_size))
        if augment_data:
            for direction in [-1,1]:
                for shift in range(1,12):
                    for length in [0, 0.5]:
                        for silence_length in [0, 0.3]:
                            train_X_temp=(train_X_new[i]).clone()
                            train_X_temp[:,2]+=direction*shift
                            train_X_temp[:,1]+=silence_length
                            train_X_temp[:,0]+=length
                            train_X_augment.append(pad(train_X_temp, pad_size))
                            train_Y_augment.append(pad(train_Y_new[i], pad_size))
    train_X_new=torch.stack(train_X_augment)
    train_Y_new=torch.stack(train_Y_augment)
    return train_X_new, train_Y_new

def validation_set_factorize(train_X, train_Y, pad_size=880):
    train_X_new=[]
    train_Y_new=[]
    for i, target in enumerate(train_Y):
        train_X_new.append(pad(torch.from_numpy(train_X[i]), pad_size))
        train_Y_new.append(pad(torch.from_numpy(train_Y[i].reshape(-1)), pad_size))
    train_X_augment=[]
    train_Y_augment=[]
    #print(train_Y_new)
    for i, target in enumerate(train_Y_new):
        train_X_augment.append(train_X_new[i])
        train_Y_augment.append(train_Y_new[i])
    train_X_new=torch.stack(train_X_augment)
    train_Y_new=torch.stack(train_Y_augment)
    return train_X_new, train_Y_new

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def compute_acc(prediction, ground_truth):
    p0t0=0
    p0t1=0
    p1t0=0
    p1t1=0
    for i in range(ground_truth.shape[0]):
        if prediction[i]==0 and ground_truth[i]==0: p0t0+=1
        if prediction[i]==0 and ground_truth[i]==1: p0t1+=1
        if prediction[i]==1 and ground_truth[i]==0: p1t0+=1
        if prediction[i]==1 and ground_truth[i]==1: p1t1+=1
    return p0t1, p0t0, p1t1, p1t0

def pad(vector, pad, dim=0):
    pad_size=list(vector.shape)
    #print(pad_size)
    pad_size[dim]=pad-vector.size(dim)
    #print(pad_size[dim])
    if pad_size[dim]<0:
        print("FATAL ERROR: pad_size=100 not enough!")
    return torch.cat([vector, torch.zeros(*pad_size).type(vector.type())], dim=dim)

class CrossValidator:
    def __init__(self, model, compute_acc, partition=5, decoder=None, batch_size=BATCH_SIZE, epochs=10, lr=1e-2, augment_data=True):
        self.model=model
        with open("/home/yixing/cross_validator_data.pkl", "rb") as f:
            dic=pickle.load(f)
            self.data_X=dic["X"]
            self.data_Y=dic["Y"]
        self.data_size=len(self.data_X)
        self.partition=partition
        self.decoder=decoder
        self.compute_acc=compute_acc
        self.train_X=[]
        self.train_Y=[]
        self.val_X=[]
        self.val_Y=[]
        self.precision_history=[]
        self.recall_history=[]
        self.loss_history=[]
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        self.augment_data_flag=augment_data
        
    def create_data(self, part):
        train_X=[]
        train_Y=[]
        val_X=[]
        val_Y=[]
        cut=int(self.data_size/self.partition)
        for i in range(self.data_size):
            if i<cut*part or i>=cut*(part+1):
                train_X.append(np.array(self.data_X[i]))
                train_Y.append(np.array(self.data_Y[i]))
            else:
                val_X.append(np.array(self.data_X[i]))
                val_Y.append(np.array(self.data_Y[i]))
        return train_X, train_Y, val_X, val_Y

    def tensorize(self, p):
        p=np.array(p)
        p=torch.from_numpy(p).float()
        p=p.to(device)
        return p

    def compute(self):
        for i in range(self.partition):
            if i>=2:
                continue
            temptrain_X, temptrain_Y, tempval_X, tempval_Y = self.create_data(i)
            #print((temptrain_X[0]), (temptrain_Y[0]))
            self.train_X, self.train_Y=training_set_factorize(temptrain_X, temptrain_Y, augment_data=self.augment_data_flag)
            self.val_X, self.val_Y=validation_set_factorize(tempval_X, tempval_Y)
            self.val_Y=self.val_Y.long()
            print(i, "phase 1 completed")
            #create dataset
            self.train_X=torch.tensor(self.train_X)
            self.train_Y=torch.tensor(self.train_Y)
            train_set=data_utils.TensorDataset(self.train_X, self.train_Y)
            train_loader=data_utils.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True) 
            cur_model = deepcopy(self.model).to(device)
            
            ij_recall_history=[]
            ij_precision_history=[]
            for j in range(self.epochs):
                if j>=5:
                    continue
                print("epoch %i"%j)
                path='model_train_epoch'+str(j)+'_part'+str(i)+'.pt'
                cur_model.load_state_dict(torch.load(path))
                #validation
                p0t1=0
                p0t0=0
                p1t1=0
                p1t0=0
                for j in range(self.val_X.size(0)):
                    print(j, self.val_X.size())
                    val_X=torch.tensor(self.val_X[j])
                    val_Y=torch.tensor(self.val_Y[j])
                    val_X=val_X.reshape(VAL_SEQ_LEN, VAL_BATCH_SIZE, -1).float().contiguous().to(device)
                    val_Y=val_Y.reshape(VAL_SEQ_LEN,).long().contiguous().to(device)
                    scores, path=cur_model(val_X)
                    prediction=torch.from_numpy(np.array(path)).reshape(VAL_SEQ_LEN,)
                    prediction=prediction.numpy()
                    val_X=val_X.cpu().numpy()
                    val_Y=val_Y.cpu().numpy()
                    last_index=np.trim_zeros(prediction, 'b').shape[0] #This line might be buggy!!! might want to use the true size
                    output_Y=prediction[0:last_index]
                    target_Y=val_Y[0:last_index]
                    output_Y[output_Y>1]=0
                    target_Y[target_Y>1]=0
                    countp0t1, countp0t0, countp1t1, countp1t0=self.compute_acc(output_Y, target_Y)
                    p0t1+=countp0t1
                    p0t0+=countp0t0
                    p1t1+=countp1t1
                    p1t0+=countp1t0

                if((p1t1+p1t0)==0):
                    ij_precision_history.append(-1)
                else:
                    ij_precision_history.append(p1t1*1.0/(p1t1+p1t0))
                
                if((p1t1+p0t1)==0):
                    ij_recall_history.append(-1)
                else:
                    ij_recall_history.append(p1t1*1.0/(p1t1+p0t1))
            self.precision_history.append(ij_precision_history)
            self.recall_history.append(ij_recall_history)
            print(self.precision_history)
            print(self.recall_history)
        return self.precision_history, self.recall_history, self.loss_history

class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size, START_TAG, STOP_TAG, BATCH_SIZE):
        super(BiLSTM_CRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.lstm1 = nn.LSTM(input_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.lstm6 = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.lstm7 = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into the output space
        self.fc7 = nn.Linear(self.hidden_dim, self.output_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.output_size, self.output_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000
        for i in range(2, self.output_size-2):
            for j in range(2, self.output_size-2):
                if j<=i:
                    self.transitions.data[j,i]=-10000
                if j>=(i+2):
                    self.transitions.data[j,i]=-10000

        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.hidden4 = self.init_hidden()
        self.hidden5 = self.init_hidden()
        self.hidden6 = self.init_hidden()
        self.hidden7 = self.init_hidden()

    def init_hidden(self, validation=False):
        if not validation:
            return (torch.randn(2, BATCH_SIZE, self.hidden_dim // 2, device=device),
                torch.randn(2, BATCH_SIZE, self.hidden_dim // 2, device=device))
        else:
            return(torch.rand(2, VAL_BATCH_SIZE, self.hidden_dim //2, device=device), torch.randn(2, VAL_BATCH_SIZE, self.hidden_dim //2, device=device))
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((BATCH_SIZE, self.output_size), -10000., device=device)
        # START_TAG has all of the score.
        helper_index=torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
        init_alphas[helper_index, START_TAG] = 0.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.output_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[helper_index, next_tag].view(BATCH_SIZE, -1).expand(BATCH_SIZE, self.output_size)
                #print(emit_score.size(), "checking emit score")
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                #print(trans_score.size(), "checking trans_score")
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                #print(next_tag_var, next_tag)
                alphas_t.append(log_sum_exp(next_tag_var, dim=1, keepdim=True).view(BATCH_SIZE))
            forward_var = torch.cat(alphas_t, 0).reshape(self.output_size, BATCH_SIZE)
            forward_var = forward_var.transpose(0,1)
            forward_var =forward_var.contiguous()
            #print(forward_var, "forward_var")
        #print(forward_var)
        terminal_var = forward_var + self.transitions[STOP_TAG]
        #print(terminal_var)
        alpha = log_sum_exp(terminal_var, dim=1, keepdim=True).reshape(BATCH_SIZE,)
        return alpha

    def _get_lstm_features(self, sequence, validation=False):
        self.hidden1 = self.init_hidden(validation=validation)
        self.hidden2 = self.init_hidden(validation=validation)
        self.hidden3 = self.init_hidden(validation=validation)
        self.hidden4 = self.init_hidden(validation=validation)
        self.hidden5 = self.init_hidden(validation=validation)
        self.hidden6 = self.init_hidden(validation=validation)
        self.hidden7 = self.init_hidden(validation=validation)
        
        lstm_out1, self.hidden1 = self.lstm1(sequence, self.hidden1)
        lstm_in2=F.relu(lstm_out1)

        lstm_out2, self.hidden2 = self.lstm2(lstm_in2, self.hidden2)
        lstm_in3=F.relu(lstm_out2)

        lstm_out3, self.hidden3 = self.lstm3(lstm_in3+lstm_in2, self.hidden3)
        lstm_in4=F.relu(lstm_out3)

        lstm_out4, self.hidden4 = self.lstm4(lstm_in4+lstm_in3, self.hidden4)
        lstm_in5=F.relu(lstm_out4)
        
        lstm_out5, self.hidden5 = self.lstm5(lstm_in5+lstm_in4, self.hidden5)
        lstm_in6=F.relu(lstm_out5)

        lstm_out6, self.hidden6 = self.lstm6(lstm_in6+lstm_in5, self.hidden6)
        lstm_in7=F.relu(lstm_out6)

        lstm_out7, self.hidden7 = self.lstm7(lstm_in7+lstm_in6, self.hidden7)
        lstm_feats=self.fc7(lstm_out7)
        
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(BATCH_SIZE, device=device)
        tags = torch.cat([torch.tensor(np.ones((1, BATCH_SIZE))*START_TAG, dtype=torch.long, device=device), tags])
        helper_index = torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[helper_index, tags[i + 1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((VAL_BATCH_SIZE, self.output_size), -10000., device=device)
        helper_index=torch.arange(VAL_BATCH_SIZE, dtype=torch.long, device=device)
        init_vvars[helper_index,START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.output_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[helper_index,best_tag_id].view(VAL_BATCH_SIZE))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t, 0).reshape(self.output_size, VAL_BATCH_SIZE))
            forward_var = forward_var.transpose(0,1)
            forward_var = forward_var.contiguous()
            #print(forward_var)
            #print(feat)
            forward_var = (forward_var + feat).view(VAL_BATCH_SIZE, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[helper_index, best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            #print(bptrs_t)
            temp=torch.cat(bptrs_t, 0).reshape(self.output_size, VAL_BATCH_SIZE)
            temp=temp.transpose(0,1).contiguous()
            #print(temp)
            best_tag_id = temp[helper_index, best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, validation=True)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

model=BiLSTM_CRF(input_dim, hidden_dim, output_size, START_TAG, STOP_TAG, BATCH_SIZE)
validator=CrossValidator(model, compute_acc=compute_acc, batch_size=BATCH_SIZE, epochs=10, lr=1e-2, augment_data=True)
precision, recall, loss=validator.compute()
print("report", precision, recall, loss)
out_f=open("lstmcrf_validation_report.pkl", "wb")
d={"p":precision, "r": recall, "l":loss}
pickle.dump(d, out_f)
out_f.close()
