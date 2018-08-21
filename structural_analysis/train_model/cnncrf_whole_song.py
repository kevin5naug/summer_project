import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F
import pickle

device = torch.device(2 if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
SEQ_LEN=880
BATCH_SIZE=2
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

class CNNCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, START_TAG, STOP_TAG, BATCH_SIZE):
        super(CNNCRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size 
        
        self.conv1 = nn.Conv1d(3, self.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 5, padding=2)
        self.fc6 = nn.Linear(self.hidden_dim, self.output_size)
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.output_size, self.output_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000
        for i in range(2, self.output_size):
            for j in range(2, self.output_size):
                if j<=i:
                    self.transitions.data[j,i]=-10000
                if j>=(i+2):
                    self.transitions.data[j,i]=-10000
    
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
        terminal_var = forward_var + self.transitions[STOP_TAG]
        #print(terminal_var)
        alpha = log_sum_exp(terminal_var, dim=1, keepdim=True).reshape(BATCH_SIZE,)
        return alpha

    def _get_cnn_features(self, sequence):
        #need to change the shape of sequence to (BATCH_SIZE, -1, SEQ_LEN)
        sequence=sequence.permute(1,2,0).contiguous()
        conv_out1 = self.conv1(sequence)
        conv_in2 = F.relu(conv_out1)
        
        conv_out2 = self.conv2(conv_in2)
        conv_in3 = F.relu(conv_out2)

        conv_out3 = self.conv3(conv_in3+conv_in2)
        conv_in4 = F.relu(conv_out3)

        conv_out4 = self.conv4(conv_in4+conv_in3)
        conv_in5 = F.relu(conv_out4)

        conv_out5 = self.conv5(conv_in5+conv_in4)
        conv_out5 = F.relu(conv_out5)
        conv_out5 = conv_out5.permute(2,0,1).contiguous()
        x = self.fc6(conv_out5)
        
        #CRF above takes in tensor of shape (Sequence_len, BATCH_SIZE, -1)
        print(x.size(), "cnn_feats")
        return x

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(BATCH_SIZE, device=device)
        tags = torch.cat([torch.tensor(np.ones((1, BATCH_SIZE))*START_TAG, dtype=torch.long, device=device), tags])
        helper_index = torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[helper_index, tags[i + 1]]
        print(score.size())
        score = score + self.transitions[STOP_TAG, tags[-1]]
        print(score.size())
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((BATCH_SIZE, self.output_size), -10000., device=device)
        helper_index=torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
        init_vvars[helper_index,START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        #print(forward_var.size(), "before loop 0")
        #print(feats.size(), "feats passed in decode process")
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
            #print(forward_var.size(), "forward var")
            #print(feat.size(), "feat")
            forward_var = (forward_var + feat).view(BATCH_SIZE, -1)
            backpointers.append(bptrs_t)
        #print(forward_var.size(), "after loop 0")
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
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_cnn_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        print(forward_score, gold_score)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_cnn_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

with open("/home/yixing/cnn_pitch_data_processed.pkl", "rb") as f:
    dic = pickle.load(f)
    train_X = dic["X"]
    train_Y = dic["Y"]

train_X = torch.tensor(train_X[0:2])
train_Y = torch.tensor(train_Y[0:2])
train_set=data_utils.TensorDataset(train_X, train_Y)
train_loader=data_utils.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
print(len(train_X))
print(train_X[0])
CLIP = 5
input_dim=3
output_size=60
START_TAG=output_size-2
STOP_TAG=output_size-1
hidden_dim=512
print_every=1
plot_every=1
plot_losses=[]
print_loss_total=0
plot_loss_total=0

model = CNNCRF(input_dim, hidden_dim, output_size, START_TAG, STOP_TAG, BATCH_SIZE).to(device)
optimizer = optim.SGD(model.parameters(), lr=5e-3, weight_decay=5e-12)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.75)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch %i"%epoch)
    #scheduler.step()
    for i, (X_train, y_train) in enumerate(train_loader):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        X_train=X_train.permute(1,0,2).float().contiguous().to(device)
        y_train=y_train.permute(1,0).long().contiguous().to(device)
        #print(X_train, y_train)
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        # Step 3. Run our forward pass.
        loss = (model.neg_log_likelihood(X_train, y_train)).sum()/BATCH_SIZE
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
    name='cnncrf_train'+str(epoch)+'.pt'
    #torch.save(model.state_dict(), name)
    torch.save(model.state_dict(),'cnncrf_train.pt')

#scores, path=model(X_train)
#prediction=torch.from_numpy(np.array(path)).reshape(SEQ_LEN,)
#print(prediction, "prediction")
#print(y_train, "y_train")
