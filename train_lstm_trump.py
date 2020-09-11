'''
@author: Sarah Boening
basierend auf: https://github.com/ChunML/NLP/tree/master/text_generation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
from argparse import Namespace

# Alle wichtigen Parameter
flags = Namespace(
    train_file='./data/trump.txt',
    output_name='lstm_trump',
    gpu_ids=0,
    epochs=100,
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['I', 'am'],
    do_train=True,
    do_predict=True,
    predict_top_k=5,
    checkpoint_path='./output/',
    model_path='./output/model-lstm_trump-finished.pth'
)


def get_data_from_file(train_file, batch_size, seq_size):
    '''
    Liest Textdatei ein und 
    '''
    with open(train_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = text.split() # es muss hier schon alles mit Leerzeichen getrennt sein, z.B. wort , wort . ansonsten muss das hier anders verarbeitet werden

    word_counts = Counter(text) # zaehlt vorhandene Woerter
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # sortierung
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)} # ID des Worts = Platz in sortierter Liste, ID: Wort
    vocab_to_int = {w: k for k, w in int_to_vocab.items()} # Wort: ID
    n_vocab = len(int_to_vocab) # Laenge des Vocabs ist spaeter wichtig fuer Netzwerk

    print('Vocabulary size', n_vocab)
    '''
    Formt Text so, dass er spaeter einfach in Batches eingeteilt werden kann 
    '''
    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    '''
    Baut Text in Batches, also Portionen, die spaeter ins Netzwerk gegeben werden
    '''
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


class RNNModule(nn.Module):
    '''
    Hier wird das Netzwerk und sein Verhalten definiert
    '''
    
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        '''
        Netzwerk-Architektur: Definiert Layer
        '''
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        # 1. Wort-Embedding ( = Encoding)
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        # 2. EIN LSTM-Layer
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        # 3. Linearer Layer (= Decoding)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        '''
        Abfolge der Layer
        '''
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state

    def zero_state(self, batch_size):
        '''
        Initiale Null-Vektoren
        '''
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


def get_loss_and_train_op(net, lr=0.001):
    '''
    Definiert Optimierungsparameter 
    Im Grunde: Mathematik mit etwas Magie
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    '''
    Wenn Training fertig, wird hier der Text generiert
    '''
    net.eval()
    words = flags.initial_words

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    choice = torch.argmax(output[0]).item()
    words.append(int_to_vocab[choice]) # 1. neues Wort

    for _ in range(100): # 100 neue worte
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
        choice = torch.argmax(output[0]).item()
        words.append(int_to_vocab[choice])
    print(' '.join(words))


def evaluate():
    '''
    Beim realen Einsatz ist es wichtig auch eine Evaluation zu haben, wie gut das Netzwerk ist
    Hier gibt es mehrere Moeglichkeiten, die alle vom Zweck des Netzwerks abhaengen
    Bei Text-Generierung moeglich: z.B. Perplexity (siehe intrinsic evaluation, perplexity as a measure for language model quality)
    Auch moeglich: text generieren und gucken ob richtig und daraufhin dann Fehler berechnen (z.B. Cross-Validation etc.)
    Bei Interesse kann ich euch meine Perplexity-Berechnung geben, (ACHTUNG: MATHE)
    '''
    pass


def main():
    '''
    1. GPU vorbeiten (wenn verfuegbar)
    '''
    dev = 'cuda:' + str(flags.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    2. Trainingsdaten vorbeiten
    '''
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)

    if flags.do_train:
        '''
        3. Netzwerk initialisieren
        '''
        net = RNNModule(n_vocab, flags.seq_size,
                        flags.embedding_size, flags.lstm_size)
        net = net.to(device)
        '''
        4. Optimierung definieren
        '''
        criterion, optimizer = get_loss_and_train_op(net, 0.01)
    
        iteration = 0
        '''
        5. Training
        '''
        for e in range(flags.epochs):
            batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
            state_h, state_c = net.zero_state(flags.batch_size)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            for x, y in batches:
                iteration += 1
                net.train() # Training-Modus, es werden Aenderungen vorgenommen
    
                optimizer.zero_grad()
    
                x = torch.LongTensor(x).to(device)
                y = torch.LongTensor(y).to(device)
    
                logits, (state_h, state_c) = net(x, (state_h, state_c))
                loss = criterion(logits.transpose(1, 2), y) # Fehler berechnen
    
                loss_value = loss.item() 
    
                loss.backward() # Backpropagation
    
                state_h = state_h.detach()
                state_c = state_c.detach()
    
                _ = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), flags.gradients_norm)
    
                optimizer.step()
    
                if iteration % 100 == 0:
                    print('Epoch: {}/{}'.format(e, flags.epochs),
                          'Iteration: {}'.format(iteration),
                          'Loss: {}'.format(loss_value))
    
                if iteration % 200 == 0:
                    # An Checkpunkten aktuellen Stand speichern
                    torch.save(net.state_dict(),
                               os.path.join(flags.checkpoint_path, 'checkpoints/model-{}-{}.pth'.format(flags.output_name, iteration)))
        # Fertiges Netzwerk speichern
        torch.save(net, os.path.join(flags.checkpoint_path, 'model-{}-{}.pth'.format(flags.output_name, 'finished')))
    else:
        print("loading model and weights")
        net = RNNModule(n_vocab, flags.seq_size,
                        flags.embedding_size, flags.lstm_size)

        # load weights from embedding trained model
        #net.load_state_dict(torch.load(flags.model_path, map_location=device)) # load checkpoint, where only the state_dict is saved
        net = torch.load(flags.model_path, map_location=device) # load full network, map_location=device is for when training was run on different gpu (ids don't match)
        net = net.to(device)
        print("done")
    if flags.do_predict:
        predict(device, net, flags.initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5)
                            
if __name__ == '__main__':
    main()
