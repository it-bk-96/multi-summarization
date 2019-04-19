import os
import math
import numpy
import nltk
import re
porter = nltk.PorterStemmer()
human_nu = 0
system_nu = 0

class LexRank(object):
	def __init__(self):
		self.text = Preprocessing()
		self.sim = DocumentSim()

	def score(self, sentences, idfs, CM, t):

		Degree = [0 for i in sentences]
		n = len(sentences)

		for i in range(n):
			for j in range(n):
				CM[i][j] = self.sim.sim(sentences[i], sentences[j], idfs)
				Degree[i] += CM[i][j]

		for i in range(n):
			for j in range(n):
				CM[i][j] = CM[i][j] / float(Degree[i])

		L = self.PageRank(CM, n)
		normalizedL = self.normalize(L)

		for i in range(len(normalizedL)):
			score = normalizedL[i]
			sentence = sentences[i]
			sentence.setLexRankScore(score)

		return sentences

	def PageRank(self,CM, n, maxerr = .0001):
		Po = numpy.zeros(n)
		P1 = numpy.ones(n)
		M = numpy.array(CM)
		t = 0
		while (numpy.sum(numpy.abs(P1-Po)) > maxerr) and (t < 100):
			Po = numpy.copy(P1)
			t = t + 1
			P1 = numpy.matmul(Po, M)
		# 	print(numpy.sum(numpy.abs(P1-Po)))
		# print(t)
		return list(Po)

	def buildMatrix(self, sentences):

		# build our matrix
		CM = [[0 for s in sentences] for s in sentences]

		for i in range(len(sentences)):
			for j in range(len(sentences)):
				CM[i][j] = 0
		return CM

	def buildSummary(self, sentences, n):
		sentences = sorted(sentences, key=lambda x: x.getLexRankScore(), reverse=True)

		summary = [sentences[0]]
		sum_len = len(sentences[0].getStemmedWords())

		# keeping adding sentences until number of words exceeds summary length
		i = 0
		while (sum_len < n):
			i += 1
			flag = True
			for sen_sum in summary:
				if sentences[i].getStemmedWords() == sen_sum.getStemmedWords():
					flag = False
			if flag:
				summary.append(sentences[i])
				sum_len += len(sentences[i].getStemmedWords())

		global system_nu, human_nu
		system_nu += sum_len
		human_nu += n
		print(sum_len)
		return summary

	def normalize(self, numbers):
		max_number = max(numbers)
		normalized_numbers = []

		for number in numbers:
			normalized_numbers.append(number / max_number)

		return normalized_numbers

	def main(self, n, path):
		sentences = self.text.openDirectory(path)
		idfs = self.sim.IDFs(sentences)
		CM = self.buildMatrix(sentences)

		sentences = self.score(sentences, idfs, CM, 0.1)

		summary = self.buildSummary(sentences, n)

		return summary


class sentence(object):

	def __init__(self, docName, stemmedWords, OGwords):

		self.stemmedWords = stemmedWords
		self.docName = docName
		self.OGwords = OGwords
		self.wordFrequencies = self.sentenceWordFreqs()
		self.lexRankScore = None

	def getStemmedWords(self):
		return self.stemmedWords

	def getDocName(self):
		return self.docName

	def getOGwords(self):
		return self.OGwords

	def getWordFreqs(self):
		return self.wordFrequencies

	def getLexRankScore(self):
		return self.LexRankScore

	def setLexRankScore(self, score):
		self.LexRankScore = score

	def sentenceWordFreqs(self):
		wordFreqs = {}
		for word in self.stemmedWords:
			if word not in wordFreqs.keys():
				wordFreqs[word] = 1
			else:
				wordFreqs[word] = wordFreqs[word] + 1

		return wordFreqs


class Preprocessing(object):

	def processFile(self, file_path_and_name):
		try:

			f = open(file_path_and_name, 'r')
			text_0 = f.read()

			# code 2007
			text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
			text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
			text_1 = re.sub("\n</TEXT>", "", text_1)

			text_1 = re.sub("<P>", "", text_1)
			text_1 = re.sub("</P>", "", text_1)
			text_1 = re.sub("\n", " ", text_1)
			text_1 = re.sub("\"", "\"", text_1)
			text_1 = re.sub("''", "\"", text_1)
			text_1 = re.sub("``", "\"", text_1)
			text_1 = re.sub(" +", " ", text_1)
			text_1 = re.sub(" _ ", "", text_1)

			text_1 = re.sub(r"\(AP\) _", " ", text_1)
			text_1 = re.sub("&\w+;", " ", text_1)

			sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
			lines = sent_tokenizer.tokenize(text_1.strip())
			# setting the stemmer

			# preprocess line[0]

			index = lines[0].find("--")
			if index != -1:
				lines[0] = lines[0][index + 2:]
			index = lines[0].find(" _ ")
			if index != -1:
				lines[0] = lines[0][index + 3:]
			sentences = []

			for sent in lines:
				sent = sent.strip()
				OG_sent = sent[:]
				sent = sent.lower()
				line = nltk.word_tokenize(sent)

				stemmed_sentence = [porter.stem(word) for word in line]
				stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
														 and x != '(' and x != ')' and x.find('&') == -1
														 and x != '?' and x != "'" and x != '!' and x != '''"'''
														 and x != '``' and x != '--' and x != ':'
														 and x != "''" and x != "'s", stemmed_sentence))

				if (len(stemmed_sentence) <= 4):
					continue

				if stemmed_sentence:
					sentences.append(sentence(file_path_and_name, stemmed_sentence, OG_sent))

			return sentences


		except IOError:
			print('Oops! File not found', file_path_and_name)
			return [sentence(file_path_and_name, [], [])]

	def get_file_path(self, file_name):
		for root, dirs, files in os.walk(os.getcwd()):
			for name in files:
				if name == file_name:
					return os.path.join(root, name)
		print("Error! file was not found!!")
		return ""

	def get_all_files(self, path=None):
		retval = []

		if path == None:
			path = os.getcwd()

		for root, dirs, files in os.walk(path):
			for name in files:
				retval.append(os.path.join(root, name))
		return retval

	def openDirectory(self, path=None):
		file_paths = self.get_all_files(path)
		sentences = []
		for file_path in file_paths:
			sentences = sentences + self.processFile(file_path)

		return sentences


class DocumentSim(object):
	def __init__(self):
		self.text = Preprocessing()

	def TFs(self, sentences):

		tfs = {}
		for sent in sentences:
			wordFreqs = sent.getWordFreqs()

			for word in wordFreqs.keys():
				if tfs.get(word, 0) != 0:
					tfs[word] = tfs[word] + wordFreqs[word]
				else:
					tfs[word] = wordFreqs[word]
		return tfs

	def TFw(self, word, sentence):
		return sentence.getWordFreqs().get(word, 0)

	def IDFs(self, sentences):

		N = len(sentences)
		idfs = {}
		words = {}
		w2 = []

		for sent in sentences:
			for word in sent.getStemmedWords():
				if sent.getWordFreqs().get(word, 0) != 0:
					words[word] = words.get(word, 0) + 1

		for word in words:
			n = words[word]
			try:
				w2.append(n)
				idf = math.log10(float(N) / n)
			except ZeroDivisionError:
				idf = 0

			idfs[word] = idf

		return idfs

	def IDF(self, word, idfs):
		return idfs[word]

	def sim(self, sentence1, sentence2, idfs):

		numerator = 0
		denom1 = 0
		denom2 = 0

		for word in sentence2.getStemmedWords():
			numerator += self.TFw(word, sentence2) * self.TFw(word, sentence1) * self.IDF(word, idfs) ** 2

		for word in sentence1.getStemmedWords():
			denom2 += (self.TFw(word, sentence1) * self.IDF(word, idfs)) ** 2

		for word in sentence2.getStemmedWords():
			denom1 += (self.TFw(word, sentence2) * self.IDF(word, idfs)) ** 2

		try:
			return numerator / (math.sqrt(denom1) * math.sqrt(denom2))

		except ZeroDivisionError:
			return float("-inf")


if __name__ == '__main__':
	root_directory = "/home/giangvu/Desktop/multi-summarization/"

	lexRank = LexRank()
	doc_folders = os.listdir(root_directory + "Data/DUC_2007/Documents")
	total_summary = []
	for folder in doc_folders:
		path = os.path.join(root_directory + "Data/DUC_2007/Documents/", '') + folder
		print("Running LexRank Summarizer for files in folder: ", folder)

		doc_summary = []
		summary = lexRank.main(250, path)
		for sentences in summary:
			text_append = re.sub("\n", "", sentences.getOGwords())
			text_append = text_append + " "
			doc_summary.append(text_append)

		results_folder = root_directory + "Data/DUC_2007/LexRank_results"
		with open(os.path.join(results_folder, (str(folder) + ".LexRank")), "w") as fileOut:
			fileOut.write("\n".join(doc_summary))

	print(system_nu, human_nu)