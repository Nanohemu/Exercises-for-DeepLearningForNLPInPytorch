# Augmenting the LSTM part-of-speech tagger with character-level features

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
ix_to_char = {}
char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            ix_to_char[len(word_to_ix)] = word
            word_to_ix[word] = len(word_to_ix)
        for c in word:
            if c not in char_to_ix:
                char_to_ix[c] = len(char_to_ix)
print(char_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = [3, 6]
HIDDEN_DIM = [3, 6]
VOCAB_SIZE = [len(char_to_ix), len(word_to_ix)]


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embeddings = nn.Embedding(vocab_size[0], embedding_dim[0])
        self.word_embeddings = nn.Embedding(vocab_size[1], embedding_dim[1])
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.char_lstm = nn.LSTM(embedding_dim[0], hidden_dim[0])
        self.word_lstm = nn.LSTM(hidden_dim[0] + embedding_dim[1], hidden_dim[1])
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim[1], tagset_size)
        self.char_hidden = None
        self.word_hidden = None
        self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.char_hidden = (torch.zeros(1, 1, self.hidden_dim[0]),
                            torch.zeros(1, 1, self.hidden_dim[0]))
        self.word_hidden = (torch.zeros(1, 1, self.hidden_dim[1]),
                            torch.zeros(1, 1, self.hidden_dim[1]))

    def forward(self, sentence):
        full_embeds = []
        for word in sentence:
            word_embed = self.word_embeddings(word).view(1, 1, -1)
            chars = ix_to_char[word.item()]
            char_seq = prepare_sequence(chars, char_to_ix)
            char_embeds = self.char_embeddings(char_seq).view(char_seq.size(0), 1, -1)
            char_lstm_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
            full_embed = [word_embed, char_lstm_out[-1]]
            embed = torch.cat(full_embed, -1)
            full_embeds.append(embed)
        embeds = torch.cat(full_embeds, 0).view(sentence.size(0), 1, -1)
        word_lstm_out, self.word_hidden = self.word_lstm(embeds, self.word_hidden)
        tag_space = self.hidden2tag(word_lstm_out.view(sentence.size(0), -1))
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out
        # before each instance
        optimizer.zero_grad()
        # Also, we need to clear out the hidden state of the LSTM, detaching it from its
        # history on the last instance.
        model.init_hidden()
        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple". i,j corresponds to score for tag j for word i.
# The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(tag_scores)
