from pickle import load, dump
from numpy import array
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

def load_file(path):
	mylist = []
	with open(path) as f:
		mylist = [line.rstrip('\n') for line in f]
	return mylist
  
# string tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# ambil max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
 
# padding
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
# one hot vector 
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
 
# NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	#Embedding(size vocab/uniq word, dimensi word vector, length input data)
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model
 
source_tr = load_file('en-id.en')
target_tr = load_file('en-id.id')
 
# prepare training data
eng_tokenizer = create_tokenizer(source_tr)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(source_tr)
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
su_tokenizer = create_tokenizer(target_tr)
su_vocab_size = len(su_tokenizer.word_index) + 1
su_length = max_length(target_tr)
print('Indonesia Vocabulary Size: %d' % su_vocab_size)
print('Indonesia Max Length: %d' % (su_length)) 
trainX = encode_sequences(su_tokenizer, su_length, target_tr)
trainY = encode_sequences(eng_tokenizer, eng_length, source_tr)
trainY = encode_output(trainY, eng_vocab_size)
print("Ready to Train")

# define model
model = define_model(su_vocab_size, eng_vocab_size, su_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(trainX, trainY), callbacks=[checkpoint], verbose=2)