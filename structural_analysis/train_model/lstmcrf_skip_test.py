import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F

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
'''
def log_sum_exp(vec):
    print(vec.size())
    max_score = vec[argmax(vec)]
    max_score_broadcast = max_score.view(BATCH_SIZE, -1).expand(BATCH_SIZE, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
'''

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


        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.hidden4 = self.init_hidden()
        self.hidden5 = self.init_hidden()
        self.hidden6 = self.init_hidden()
        self.hidden7 = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, BATCH_SIZE, self.hidden_dim // 2, device=device),
                torch.randn(2, BATCH_SIZE, self.hidden_dim // 2, device=device))

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

    def _get_lstm_features(self, sequence):
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.hidden4 = self.init_hidden()
        self.hidden5 = self.init_hidden()
        self.hidden6 = self.init_hidden()
        self.hidden7 = self.init_hidden()
        
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
        init_vvars = torch.full((BATCH_SIZE, self.output_size), -10000., device=device)
        helper_index=torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
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
                viterbivars_t.append(next_tag_var[helper_index,best_tag_id].view(BATCH_SIZE))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t, 0).reshape(self.output_size, BATCH_SIZE))
            forward_var = forward_var.transpose(0,1)
            forward_var = forward_var.contiguous()
            #print(forward_var)
            #print(feat)
            forward_var = (forward_var + feat).view(BATCH_SIZE, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[helper_index, best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            #print(bptrs_t)
            temp=torch.cat(bptrs_t, 0).reshape(self.output_size, BATCH_SIZE)
            temp=temp.transpose(0,1).contiguous().numpy()
            #print(temp)
            best_tag_id = temp[helper_index, best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start[0] == START_TAG  # Sanity check
        best_path.reverse()
        best_path=np.concatenate(best_path, 0).reshape(SEQ_LEN, BATCH_SIZE, -1)
        best_path=np.transpose(best_path, (1,0,2))
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

import pickle
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
# load data from file
SEQ_LEN=80
BATCH_SIZE=2
with open("/home/yixing/pitch_data_processed.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]

train_X = torch.tensor(train_X)
train_Y = torch.tensor(train_Y)
train_set=data_utils.TensorDataset(train_X[0:BATCH_SIZE], train_Y[0:BATCH_SIZE])
train_loader=data_utils.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False)

# In[92]:


print(len(train_X))
#print(train_X[0])

CLIP = 10
input_dim=2
output_size=60
START_TAG=output_size-2
STOP_TAG=output_size-1
hidden_dim=512
print_every=50
plot_every=50
plot_losses=[]
print_loss_total=0
plot_loss_total=0
'''
seq1=[[[1,2,60]], [[2,5,72]], [[5,9,62]], [[9,10,66]], [[10,17,70]], [[17, 20, 67]]]
data1=torch.tensor(seq1, dtype=torch.float)
truth1=[0,2,0,1,1,2]
label1=torch.tensor(truth1, dtype=torch.long)
'''
model = BiLSTM_CRF(input_dim, hidden_dim, output_size, START_TAG, STOP_TAG, BATCH_SIZE).to(device)
model.load_state_dict(torch.load('lstmcrf_train.pt'))
model.eval()
for i, (X_train, y_train) in enumerate(train_loader):
     print(X_train, y_train)
     #X_train=X_train.reshape(SEQ_LEN,BATCH_SIZE,input_dim).float().to(device)
     X_train=X_train.transpose(0,1).float().contiguous().to(device)
     y_train=y_train.transpose(0,1).long().contiguous().to(device)
     if(i>5):
        break
     else:
        scores, path=model(X_train)
        print(X_train, "X_train")
        print(scores, "scores")
        #prediction=path.transpose(0,1).cpu().long().contiguous()
        #print(model(X_train), "hello")
        prediction=torch.from_numpy(path)
        print(prediction, "prediction")
        print(y_train, "y_train")
# We got it!
