from pyvi import ViTokenizer
import nltk

def split_sentences(file_name):
	with open(file_name, 'r') as file:
		text_system = file.read()

	sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
	tmp = sentence_token.tokenize(text_system)

	sentences = []
	for item in tmp:
		if "…" in item:
			b = item.split("…")
			for i in b:
				sentences.append(i)
		else:
			sentences.append(item)

	return sentences

def get_all_sentences(file_system, file_reference):
	sentences_origin_system = []
	for item in file_system:
		sentences_origin_system += split_sentences(item)

	sentences_system = []
	for item in sentences_origin_system:
		sentences_system.append(ViTokenizer.tokenize(item))

	sentences_origin_reference = []
	for item in file_reference:
		with open(item, 'r') as file:
			sentences_origin_reference.append(file.read())

	# for item in sentences_origin_reference:
	# 	print (item)

	sentences_reference = []
	for item in sentences_origin_reference:
		sentences_reference.append(ViTokenizer.tokenize(item))

	return sentences_origin_system, sentences_system, sentences_reference