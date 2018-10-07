import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size, START_TAG, STOP_TAG):
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
        return (torch.randn(2, 1, self.hidden_dim // 2, device=device),
                torch.randn(2, 1, self.hidden_dim // 2, device=device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.output_size), -10000., device=device)
        # START_TAG has all of the score.
        init_alphas[0][START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.output_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[0, next_tag].view(1, -1).expand(1, self.output_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[STOP_TAG]
        alpha = log_sum_exp(terminal_var)
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
        #todo: test Batchnorm 1d layer

        lstm_out2, self.hidden2 = self.lstm2(lstm_out1, self.hidden2)
        
        lstm_out3, self.hidden3 = self.lstm3(lstm_out2+lstm_out1, self.hidden3)

        lstm_out4, self.hidden4 = self.lstm4(lstm_out3+lstm_out2, self.hidden4)
        
        lstm_out5, self.hidden5 = self.lstm5(lstm_out4+lstm_out3, self.hidden5)
        
        lstm_out6, self.hidden6 = self.lstm6(lstm_out5+lstm_out4, self.hidden6)
        
        lstm_out7, self.hidden7 = self.lstm7(lstm_out6+lstm_out5, self.hidden7)
        lstm_feats=self.fc7(lstm_out7)
        
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=device)
        tags = torch.cat([torch.tensor([START_TAG], dtype=torch.long, device=device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[0, tags[i + 1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.output_size), -10000., device=device)
        init_vvars[0][START_TAG] = 0

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
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
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
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load data from file
with open("pitch_data.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]

def input_transform(train_x, time_x, i):
    output = torch.from_numpy(np.array([train_x[i], time_x[i]]))
    return output.transpose(1, 0).to(device)

def input_factorize(train_x):
    output = []
    for i in range(train_x.shape[0]):
        content=np.array_split(train_x[i], train_x[i].shape[0]/9)
        for index in range(len(content)):
            if (len(content[index]))<10:
                output.append(content[index])
    return output


def target_factorize(train_y):
    output = []
    for i in range(train_y.shape[0]):
        content=np.array_split(train_y[i], train_y[i].shape[0]/9)
        for index in range(len(content)):
            if (len(content[index]))<10:
                output.append(content[index])
    return output

def target_transform(train_y):
    output = torch.zeros((1, 2))
    output[0, int(train_y)] = 1
    return output.unsqueeze(1).to(device)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

train_X = input_factorize(train_X)
train_X = torch.tensor(train_X)
train_Y = torch.tensor(target_factorize(train_Y))
train_set=data_utils.TensorDataset(train_X, train_Y)
train_loader=data_utils.DataLoader(dataset=train_set, shuffle=True)

# In[92]:


print(len(train_X))
print(train_X[0])

START_TAG = 5
STOP_TAG = 6
input_dim=3
output_size=7
hidden_dim=512
print_every=100
plot_every=100
plot_losses=[]
print_loss_total=0
plot_loss_total=0

# Make up some training data
'''
seq1=[[[1,2,60]], [[2,5,72]], [[5,9,62]], [[9,10,66]], [[10,17,70]], [[17, 20, 67]]]
data1=torch.tensor(seq1, dtype=torch.float)
truth1=[0,2,0,1,1,2]
label1=torch.tensor(truth1, dtype=torch.long)
'''
model = BiLSTM_CRF(input_dim, hidden_dim, output_size, START_TAG, STOP_TAG).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch %i"%epoch)
    for i, (X_train, y_train) in enumerate(train_loader):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        X_train=X_train.reshape(9,1,3).float().to(device)
        y_train=y_train.reshape(9,).long().to(device)
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(X_train, y_train)
        print_loss_total+=loss
        plot_loss_total+=loss
        
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %.4f)' % (i, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
model.save_state_dict('lstmcrf_train.pt')
showPlot(plot_losses)

# We got it!
