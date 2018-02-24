import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import random
import sys
import os
import codecs
import collections
from six.moves import cPickle


def get_simple_data():
    """
    Reads and concatenates all comments
    """
    out = ''
    NUM_COMMENTS = 1000
    counter = 0
    with open('comments.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            counter += 1
            text  = r[3]
            out += text
            if counter == NUM_COMMENTS:
                break
    return out


def train():
    data_dir = 'data/Artistes_et_Phalanges-David_Campion'# data directory containing input.txt
    save_dir = 'save' # directory to store models
    rnn_size = 128 # size of RNN
    batch_size = 30 # minibatch size
    seq_length = 15 # sequence length
    num_epochs = 1 # number of epochs
    learning_rate = 0.001 #learning rate
    sequences_step = 1 #step to create sequences

    # input_file = os.path.join(data_dir, "input.txt")
    vocab_file = os.path.join(save_dir, "words_vocab.pkl")
    
    data = get_simple_data()
    x_text = data.split()
    word_counts = collections.Counter(x_text)

    # Mapping from index to word : that's the vocabulary
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocabulary_inv)}
    words = [x[0] for x in word_counts.most_common()]

    #size of the vocabulary
    vocab_size = len(words)
    
    #save the words and vocabulary
    with open(os.path.join(vocab_file), 'wb') as f:
        cPickle.dump((words, vocab, vocabulary_inv), f)

    #create sequences
    sequences = []
    next_words = []
    
    for i in range(0, len(x_text) - seq_length, sequences_step):
        sequences.append(x_text[i: i + seq_length])
        next_words.append(x_text[i + seq_length])
        print 'nb sequences:' + str(len(sequences))

    print 'vecotrization'
    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, word in enumerate(sentence):
            X[i, t, vocab[word]] = 1
        y[i, vocab[next_words[i]]] = 1

    # build the model: a single LSTM
    print('Build LSTM model.')
    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=(seq_length, vocab_size)))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    #adam optimizer
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # fit the model
    # TODO: this should use fit_generator (and have a generator)
    model.fit(X, y,batch_size=batch_size,epochs=num_epochs)

    #save the model
    model.save(save_dir + "/" + 'my_model.h5')    


# TODO: temperature...?
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate():
    save_dir = 'save' # directory where model is stored
    batch_size = 15 # minibatch size
    seq_length = 15 # sequence length
    words_number = 400 #number of words to generate
    seed_sentences = "the" #sentence for seed generation
    vocab_file = os.path.join(save_dir, "words_vocab.pkl")

    with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

    vocab_size = len(words)
    # load the model
    model = load_model(save_dir + '/' + 'my_model.h5')
    generated = ''
    sentence = []
    for i in range (seq_length):
        sentence.append("a")

    seed = seed_sentences.split()
    for i in range(len(seed)):
        sentence[seq_length-i-1] = seed[len(seed)-i-1]

    generated += ' '.join(sentence)

    for i in range(words_number):
        #create the vector
        x = np.zeros((1, batch_size, vocab_size))
        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1.

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 1)
        next_word = vocabulary_inv[next_index]

        #add the next word to the text
        generated += " " + next_word
        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]

    print generated


def main():
    # train()
    generate()


if __name__ == '__main__':
    main()
