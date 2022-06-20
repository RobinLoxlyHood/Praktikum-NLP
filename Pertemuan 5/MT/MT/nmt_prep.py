import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
 
# load dokumen ke memory
def load_doc(filename):
	# read only
	file = open(filename, mode='rt', encoding='utf-8')
	# baca semua teks
	text = file.read()
	# close file
	file.close()
	return text
 
# split dokumen ke kata
def to_pairs(doc):
	#split newline
	lines = doc.strip().split('\n')
	#untuk setiap line kita split tab (en)
	pairs = [line.split('\t') for line in  lines]
	return pairs
 
# clean list
def clean_pairs(lines):
	cleaned = list()
	# filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# hapus punctuation, jadikan dalam table 
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize
			line = line.split()
			# lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)
 
# save kalimat bersih ke file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)
 
# load dataset
filename = 'en-id.txt'
doc = load_doc(filename)
# split into english-indonesia pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs ke file
save_clean_data(clean_pairs, 'english-indonesia.pkl')
# cek 100 pair pertama
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))