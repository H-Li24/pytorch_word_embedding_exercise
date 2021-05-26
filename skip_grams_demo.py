import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 4  # 左右各2个单词
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# 通过从 `raw_text` 得到一组单词, 进行去重操作
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = raw_text[i]
    target = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    data.append((context, target))
# print(data[:5])


class SkipgramModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SkipgramModeler, self).__init__()
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, context_size * vocab_size)
    
    def forward(self,x):
        embeds = self.embeddings(x).view(1,-1)
        output = self.linear1(embeds)
        output = F.relu(output)
        output = self.linear2(output)
        log_probs = F.log_softmax(output, dim=1).view(self.context_size, -1)
        return log_probs

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[context]]
    return torch.tensor(idxs, dtype=torch.long)

def make_target_vectors(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

losses = []
loss_function = nn.NLLLoss()
model = SkipgramModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = 0
    for context, target in data:
        context_id = make_context_vector(context, word_to_ix)

        model.zero_grad()

        log_probs = model(context_id)

        loss_0 = loss_function(log_probs[0].unsqueeze(0), torch.tensor([word_to_ix[target[0]]], dtype=torch.long))
        loss_1 = loss_function(log_probs[1].unsqueeze(0), torch.tensor([word_to_ix[target[1]]], dtype=torch.long))
        loss_2 = loss_function(log_probs[2].unsqueeze(0), torch.tensor([word_to_ix[target[2]]], dtype=torch.long))
        loss_3 = loss_function(log_probs[3].unsqueeze(0), torch.tensor([word_to_ix[target[3]]], dtype=torch.long))
        loss = loss_0 + loss_1 + loss_2 + loss_3

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

correct = 0
for context, target in data:
    context_idxs = torch.tensor(make_context_vector(context, word_to_ix), dtype=torch.long)
    log_probs = model(context_idxs)
    _, ix = torch.max(log_probs, 1)
    target_ix = make_target_vectors(target, word_to_ix)

    target_ix_1 = list(target_ix)
    ix_1 = list(ix)
    retA = [i for i in ix_1 if i in target_ix_1]
    if len(retA) != 0:
        correct += 1
    
    # correct += (target_1 == ix_1)

accuracy = correct / (len(data))
print("Average accuracy:", accuracy)