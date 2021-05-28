import json
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

import preprocess
import model_lstm

n = 16 #2                    # Number of words used in prediction
min_occurences = 0 #1       # Minimum number of occurences of a word for it to occur in vocabulary
batch_size = 32 #1

text = preprocess.load_text()

word_to_id, id_to_word = preprocess.get_vocab(text, min_occurences)

ids =  [word_to_id[word] for word in text]


training_dataset = preprocess.get_tensor_dataset(ids, n)
training_loader = DataLoader(training_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

# Size parameters
vocab_size = len(word_to_id) + 1
embedding_dim = 256     # size of the word embeddings
hidden_dim = 256        # size of the hidden state
n_layers = 2            # number of LSTM layers

# Training parameters
epochs = 14 #10
learning_rate = 0.001
clip = 1

net = model_lstm.LSTM(vocab_size, embedding_dim, hidden_dim, n_layers)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

net.train()
for e in range(epochs):
    tl = 0
    hidden = net.init_hidden(batch_size)

    # loops through each batch
    for features, labels in training_loader:


        # resets training history
        hidden = tuple([each.data for each in hidden])
        net.zero_grad()
        # computes gradient of loss from backprop
        output, hidden = net.forward(features, hidden)
        loss = loss_func(output, labels)
        loss.backward()
        tl += loss
        
        # using clipping to avoid exploding gradient
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        
    print("Epoch [{}]".format(e+1) + "  Loss: {}".format(tl/batch_size))
   

net.eval()
torch.save(net, 'model/trained_model.pt')

with open('model/word_to_id.json', 'w') as fp:
    json.dump(word_to_id, fp, indent=4)
