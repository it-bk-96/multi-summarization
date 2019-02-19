import os
import math
# import sentence
import nltk
from pyvi import ViTokenizer
import numpy as np
import time
from sklearn.decomposition import NMF


class sentence(object):

	# ------------------------------------------------------------------------------
	# Description	: Constructor to initialize the setence object
	# Parameters  	: docName, name of the document/file
	#				  preproWords, words of the file after the stemming process
	#				  originalWords, actual words before stemming
	# Return 		: None
	# ------------------------------------------------------------------------------
	def __init__(self, docName, preproWords, originalWords):
		self.docName = docName
		self.preproWords = preproWords
		self.wordFrequencies = self.sentenceWordFreq()
		self.originalWords = originalWords
		self.score = 0

	def setScore(self, score):
		self.score = score

	def getScore(self):
		return self.score

	# ------------------------------------------------------------------------------
	# Description	: Function to return the name of the document
	# Parameters	: None
	# Return 		: name of the document
	# ------------------------------------------------------------------------------
	def getDocName(self):
		return self.docName

	# ------------------------------------------------------------------------------
	# Description	: Function to return the stemmed words
	# Parameters	: None
	# Return 		: stemmed words of the sentence
	# ------------------------------------------------------------------------------
	def getPreProWords(self):
		return self.preproWords

	# ------------------------------------------------------------------------------
	# Description	: Function to return the original words of the sentence before
	#				  stemming
	# Parameters	: None
	# Return 		: pre-stemmed words
	# ------------------------------------------------------------------------------
	def getOriginalWords(self):
		return self.originalWords

	# ------------------------------------------------------------------------------
	# Description	: Function to return a dictonary of the word frequencies for
	#				  the particular sentence object
	# Parameters	: None
	# Return 		: dictionar of word frequencies
	# ------------------------------------------------------------------------------
	def getWordFreq(self):
		return self.wordFrequencies

	# ------------------------------------------------------------------------------
	# Description	: Function to create a dictonary of word frequencies for the
	#				  sentence object
	# Parameters	: None
	# Return 		: dictionar of word frequencies
	# ------------------------------------------------------------------------------
	def sentenceWordFreq(self):
		wordFreq = {}
		for word in self.preproWords:
			if word not in wordFreq.keys():
				wordFreq[word] = 1
			else:
				wordFreq[word] += + 1

		return wordFreq
# nltk.download('punkt')

# ---------------------------------------------------------------------------------
# Description	: Function to preprocess the files in the document cluster before
#				  passing them into the NFM summarizer system. Here the sentences
#				  of the document cluster are modelled as sentences after extracting
#				  from the files in the folder path. 
# Parameters	: file_name, name of the file in the document cluster
# Return 		: list of sentence object
# ---------------------------------------------------------------------------------

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
			sent = sentence(file_name, stemmedSent, originalWords)
			sentences.append(sent)


	return sentences


# ---------------------------------------------------------------------------------
# Description	: Function to find the term frequencies of the words in the
#				  sentences present in the provided document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, term frequency score
# ---------------------------------------------------------------------------------
def TFs(sentences):
	# initialize tfs dictonary
	tfs = {}

	# for every sentence in document cluster
	for sent in sentences:
		# retrieve word frequencies from sentence object
		wordFreqs = sent.getWordFreq()

		# for every word
		for word in wordFreqs.keys():
			# if word already present in the dictonary
			if tfs.get(word, 0) != 0:
				tfs[word] = tfs[word] + wordFreqs[word]
			# else if word is being added for the first time
			else:
				tfs[word] = wordFreqs[word]
	return tfs


# ---------------------------------------------------------------------------------
# Description	: Function to find the inverse document frequencies of the words in
#				  the sentences present in the provided document cluster 
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, inverse document frequency score
# ---------------------------------------------------------------------------------
def IDFs(sentences):
	N = len(sentences)
	idfs = {}
	words = {}
	w2 = []
	# every sentence in our cluster
	for sent in sentences:

		# every word in a sentence
		for word in sent.getPreProWords():
			# not to calculate a word's IDF value more than once
			if sent.getWordFreq().get(word, 0) != 0:
				words[word] = words.get(word, 0) + 1

	# for each word in words
	for word in words:
		n = words[word]

		# avoid zero division errors
		try:
			w2.append(n)
			idf = math.log10(float(N) / n)
		except ZeroDivisionError:
			idf = 0

		# reset variables
		idfs[word] = idf

	return idfs


# ---------------------------------------------------------------------------------
# Description	: Function to find TF-IDF score of the words in the document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, TF-IDF score
# ---------------------------------------------------------------------------------
def TF_IDF(sentences):
	# Method variables
	tfs = TFs(sentences)
	idfs = IDFs(sentences)
	retval = {}

	# for every word
	for word in tfs:
		# calculate every word's tf-idf score
		tf_idfs = tfs[word] * idfs[word]

		# add word and its tf-idf score to dictionary
		if retval.get(tf_idfs, None) == None:
			retval[tf_idfs] = [word]
		else:
			retval[tf_idfs].append(word)

	return retval


def makeSummary(sentences, n, scores):

	sentences = sorted(sentences, key=lambda x: x.getScore(), reverse=True)
	summary = []
	i = 0
	length_summary = len(ViTokenizer.tokenize(sentences[i].getOriginalWords().strip()).split())
	while (length_summary < n):
		i += 1
		summary += [sentences[i]]
		length_summary += len(ViTokenizer.tokenize(sentences[i].getOriginalWords().strip()).split())

	return summary


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == '__main__':

	# set the main Document folder path where the subfolders are present
	main_folder_path = os.getcwd() + "/Data_Chưa_tách_từ/Documents"
	human_folder_path = os.getcwd() + "/Data_Chưa_tách_từ/Human_Summaries/"

	stop_word = list(map(lambda x: "_".join(x.split()), open("/home/giangvu/Desktop/multi-summarization/vietnamese-stopwords.txt", 'r').read().split("\n")))

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

		# calculate TF, IDF and TF-IDF scores
		# TF_w 		= TFs(sentences)
		IDF_w = IDFs(sentences)
		TF_IDF_w = TF_IDF(sentences)
		vocabulary = []
		for sent in sentences:
			vocabulary = vocabulary + sent.getPreProWords()
		vocabulary = list(set(vocabulary))
		A = np.zeros(shape=(len(vocabulary), len(sentences)))

		for i in range(len(sentences)):
			tf_sentence = sentences[i].getWordFreq()
			for word in tf_sentence.keys():
				index = vocabulary.index(word)
				A[index][i] += tf_sentence[word]

		rank_A = np.linalg.matrix_rank(A)
		print(rank_A)
		model = NMF(n_components=rank_A, init='random', random_state=0)
		W = model.fit_transform(A)
		H = model.components_

		scores = np.sum(H, axis=0)
		for i in range(len(sentences)):
			sentences[i].setScore(scores[i])

		# build summary
		summary = makeSummary(sentences, length_summary, scores)

		final_summary = ""
		for sent in summary:
			final_summary = final_summary + sent.getOriginalWords() + "\n"
		final_summary = final_summary[:-1]
		results_folder = os.getcwd() + "/Data_Chưa_tách_từ/NMF_results"
		with open(os.path.join(results_folder, (str(folder) + ".NMF")), "w") as fileOut:
			fileOut.write(final_summary)
		print("Execution time: " + str(time.time() - start_time))
