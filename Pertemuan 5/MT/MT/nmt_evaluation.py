from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
 
# load data bersih
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# max panjang kalimat
def max_length(lines):
	return max(len(line.split()) for line in lines)
 
# padding
def encode_sequences(tokenizer, length, lines):
	# ubah ke integer sequence
	X = tokenizer.texts_to_sequences(lines)
	# pading
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
# reverse dari vector ke word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# P(target|source) generate
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	#hasil prediksi dari one hot vector di argmax
	integers = [argmax(vector) for vector in prediction]
	target = list()
	#loop setiap data prediksi
	for i in integers:
		#convert ke target word
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
# evaluasi kemampuan model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src, test  = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
 
# load datasets
dataset = load_clean_sentences('english-indonesia-both.pkl')
train = load_clean_sentences('english-indonesia-train.pkl')
test = load_clean_sentences('english-indonesia-test.pkl')
# english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# indonesia tokenizer
id_tokenizer = create_tokenizer(dataset[:, 1])
id_vocab_size = len(id_tokenizer.word_index) + 1
id_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(id_tokenizer, id_length, train[:, 1])
testX = encode_sequences(id_tokenizer, id_length, test[:, 1])
 
# load model
model = load_model('model.h5')
# test on some training sequences
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX, test)