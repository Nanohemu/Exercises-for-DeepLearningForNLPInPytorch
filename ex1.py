# Computing Word Embeddings: Continuous Bag-of-Words

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process. Computational processes are abstract
beings that inhabit computers. As they evolve, processes manipulate other abstract
things called data. The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
vocab = set(raw_text)
word_to_ix = {word: i for i, word in enumerate(set(raw_text))}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

print(data[:5])


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.context = 2 * context_size  # words before and after the target
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        emb = self.embed(inputs).view(self.context, -1)
        emb = torch.sum(emb, dim=0)
        out = self.fc1(emb)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        log_probs = F.log_softmax(out, dim=0)
        return log_probs


# create your model and train. here are some functions to help you make the data ready for use by your module
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return tensor


loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
print(make_context_vector(data[0][0], word_to_ix))  # example

for epoch in range(50):
    total_loss = 0.
    for context, target in data:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_var = make_context_vector(context, word_to_ix)
        label = torch.LongTensor([word_to_ix[target]])
        # Step 2. Recall that torch *accumulates* gradients. Before passing in a new instance,
        # you need to zero out the gradients from the old instance
        optimizer.zero_grad()
        # Step 3. Run the forward pass, getting log probabilities over next words
        log_probs = model(context_var).view(1, len(vocab))
        # Step 4. Compute your loss function. (Again, Torch wants the target word wrapped in a variable)
        loss = loss_function(log_probs, label)
        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(total_loss)
