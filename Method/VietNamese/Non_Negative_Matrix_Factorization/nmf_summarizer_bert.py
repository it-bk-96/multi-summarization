import os
# import sentence
import nltk
from pyvi import ViTokenizer
import numpy as np
import time

from sklearn.decomposition import NMF
from bert import extract_features

root_directory = "/home/giangvu/Desktop/multi-summarization/"

class sentence(object):

	def __init__(self, docName, originalWords):
		self.docName = docName
		self.originalWords = originalWords
		self.score = 0
		self.vectorSentence = []


	def getVectorSentence(self):
		return self.vectorSentence

	def setVectorSentence(self, vector):
		self.vectorSentence = vector

	def setScore(self, score):
		self.score = score

	def getScore(self):
		return self.score

	def getDocName(self):
		return self.docName

	def getOriginalWords(self):
		return self.originalWords


def processFile(file_name):
	# read file from provided folder path
	f = open(file_name, 'r')
	text_1 = f.read()

	# tách câu
	sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
	lines = sentence_token.tokenize(text_1.strip())

	# setting the stemmer
	sentences = []

	# modelling each sentence in file as sentence object
	for i in range(len(lines)):
		line = lines[i]
		# giữ lại câu gốc
		originalWords = line[:]

		# chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
		line = line.strip().lower()

		# tách từ
		stemmedSent = ViTokenizer.tokenize(line).split()

		stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
		                               and x != '!' and x != '''"''' and x != "''" and x != '-' and x not in stop_word, stemmedSent))

		if ((i + 1) == len(lines)) and (len(stemmedSent) <= 5):
			break
		# list of sentence objects
		if stemmedSent != []:
			sent = sentence(file_name, originalWords)
			sentences.append(sent)


	return sentences

def makeSummary(sentences, n):
	scores = np.sum(H, axis=0)
	for i in range(len(sentences)):
		sentences[i].setScore(scores[i])
	sentences = sorted(sentences, key=lambda x: x.getScore(), reverse=True)
	summary = []
	i = 0
	length_summary = len(ViTokenizer.tokenize(sentences[i].getOriginalWords().strip()).split())
	while (length_summary < n):
		i += 1
		summary += [sentences[i]]
		length_summary += len(ViTokenizer.tokenize(sentences[i].getOriginalWords().strip()).split())

	return summary

def makeSummary_Gong_Liu(sentences, n, H):

	i = 0
	index_H = np.argmax(H[i])
	length_summary = len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())
	array_index = []
	array_index.append(index_H)
	while (length_summary < n):
		i += 1
		index_H = np.argmax(H[i])
		if index_H not in array_index:
			array_index.append(index_H)
			length_summary += len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())
	print(n, length_summary)

	return [sentences[index] for index in array_index]


def makeSummary_Topic(sentences, n, H):

	length_sentences = len(sentences)
	length_concept = len(H)
	average_concept = [sum(H[i][j] for j in range(length_sentences))/length_sentences for i in range(length_concept)]
	for i in range(length_concept):
		for j in range(length_sentences):
			if H[i][j] < average_concept[i]:
				H[i][j] = 0
	matrix_conceptxconcept = [[0 for _ in range(length_concept)] for _ in range(length_concept)]

	for i in range(length_concept):
		for j in range(length_concept):
			if matrix_conceptxconcept[j][i] != 0:
				matrix_conceptxconcept[i][j] = matrix_conceptxconcept[j][i]
				continue
			total = 0
			for k in range(length_sentences):
				if (H[i][k] != 0) and (H[j][k] != 0):
					total += (H[i][k] + H[j][k])
			matrix_conceptxconcept[i][j] = total

	strength_concept = [sum(matrix_conceptxconcept[i][j] for j in range(length_concept)) for i in range(length_concept)]
	top_index = np.array(strength_concept).argsort()[-length_concept:][::-1]
	i = 0
	index_H = np.argmax(H[top_index[i]])
	length_summary = len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())
	array_index = []
	array_index.append(index_H)
	while (length_summary < n):
		i += 1
		index_H = np.argmax(H[top_index[i]])
		if index_H not in array_index:
			array_index.append(index_H)
			length_summary += len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())

	return [sentences[index] for index in array_index]


if __name__ == '__main__':

	main_folder_path = root_directory + "Data/Data_VietNamese/Documents"
	human_folder_path = root_directory + "Data/Data_VietNamese/Human_Summaries/"

	stop_word = list(map(lambda x: "_".join(x.split()),
						 open(root_directory + "vietnamese-stopwords.txt", 'r').read().split(
							 "\n")))

	# read in all the subfolder names present in the main folder

	# read in all the subfolder names present in the main folder
	for folder in os.listdir(main_folder_path):
		start_time = time.time()

		print("Running NMF Summarizer for files in folder: ", folder)
		# for each folder run the MMR summarizer and generate the final summary
		curr_folder = main_folder_path + "/" + folder

		# find all files in the sub folder selected
		files = os.listdir(curr_folder)

		file_human_1 = human_folder_path + folder + ".ref1.txt"
		file_human_2 = human_folder_path + folder + ".ref2.txt"
		text_1 = open(file_human_1, 'r').read()
		text_2 = open(file_human_2, 'r').read()
		text_1_token = ViTokenizer.tokenize(text_1)
		text_2_token = ViTokenizer.tokenize(text_2)
		length_summary = int((len(text_1_token.split()) + len(text_1_token.split()))/2)

		sentences = []

		for file in files:
			sentences = sentences + processFile(curr_folder + "/" + file)

		A = np.zeros(shape=(768, len(sentences)))


		vocab_file = root_directory + 'bert/multi_cased_L-12_H-768_A-12/vocab.txt'
		bert_config_file = root_directory + 'bert/multi_cased_L-12_H-768_A-12/bert_config.json'
		layers = "-1, -2, -3, -4"
		init_checkpoint = root_directory + 'bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
		max_seq_length = 128
		batch_size = 8


		extract_features.getFeature(sentences, bert_config_file, vocab_file, init_checkpoint, layers,
													   max_seq_length, batch_size)

		for i in range(len(sentences)):
			vector = sentences[i].getVectorSentence()
			for j in range(300):
				if vector[j] < 0:
					A[j][i] = 0
				else:
					A[j][i] = vector[j]

		rank_A = np.linalg.matrix_rank(A)

		print(A)
		model = NMF(n_components=rank_A, init='random', random_state=0)
		W = model.fit_transform(A)
		H = model.components_

		# build summary
		summary = makeSummary_Gong_Liu(sentences, length_summary, H)
		# summary = makeSummary(sentences, length_summary)
		# summary = makeSummary_Topic(sentences, length_summary, H)

		final_summary = ""
		for sent in summary:
			final_summary = final_summary + sent.getOriginalWords() + "\n"
		final_summary = final_summary[:-1]
		results_folder = root_directory + "Data/Data_VietNamese/NMF_results_bert"
		with open(os.path.join(results_folder, (str(folder) + ".NMF")), "w") as fileOut:
			fileOut.write(final_summary)
		print("Execution time: " + str(time.time() - start_time))

