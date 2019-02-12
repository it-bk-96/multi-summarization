import os
import numpy
import nltk
import re

class TextRank(object):
	def __init__(self):
		self.text = Preprocessing()

	def PageRank(self, graph, node_weights, d=.85, iter=20):
		weight_sum = numpy.sum(graph, axis=0)
		while iter > 0:
			for i in range(len(node_weights)):
				temp = 0.0
				for j in range(len(node_weights)):
					temp += graph[i,j] * node_weights[j] / weight_sum[j]
				node_weights[i] = 1 - d + (d * temp)
			iter-=1

	def buildSummary(self, sentences, node_weights, n):

		summary = []
		top_index = [i for i,j in sorted(enumerate(node_weights), key=lambda x: x[1],reverse=True)[:n]]

		for i in top_index:
			summary += [sentences[i]]
		return summary

	def main(self, n, path):
		sentences = self.text.openDirectory(path)
		num_nodes = len(sentences)
		graph = numpy.zeros((num_nodes, num_nodes))

		for i in range(num_nodes):
			for j in range(i+1, num_nodes):   # tinh toan độ trùng lặp giữa 2 sentences
				graph[i,j] = float(len(set(sentences[i].getStemmedWords()) & set(sentences[j].getStemmedWords())))/(len(sentences[i].getStemmedWords()) + len(sentences[j].getStemmedWords()))
				graph[j,i] = graph[i,j]

		node_weights = numpy.ones(num_nodes)
		self.PageRank(graph, node_weights)
		summary = self.buildSummary(sentences, node_weights, n)

		return summary


class sentence(object):

	def __init__(self, stemmedWords, OGwords):

		self.stemmedWords = stemmedWords
		self.OGwords = OGwords
		self.lexRankScore = None

	def getStemmedWords(self):
		return self.stemmedWords

	def getOGwords(self):
		return self.OGwords



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

			sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
			lines = sent_tokenizer.tokenize(text_1.strip())

			sentences = []
			porter = nltk.PorterStemmer()

			for sent in lines:
				OG_sent = sent[:]
				sent = sent.strip().lower()
				line = nltk.word_tokenize(sent)

				stemmed_sentence = [porter.stem(word) for word in line]
				stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'"
				                                    and x != '!' and x != '''"''' and x != "''" and x != "'s",
				                          stemmed_sentence))
				if stemmed_sentence != []:
					sentences.append(sentence(stemmed_sentence, OG_sent))

			return sentences


		except IOError:
			print('Oops! File not found', file_path_and_name)
			return [sentence([], [])]

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

if __name__ == '__main__':

	textRank = TextRank()
	doc_folders = os.listdir("Data_DUC_2004/Documents")
	total_summary = []
	summary_length = 10

	for folder in doc_folders:
		path = os.path.join("Data_DUC_2004/Documents/", '') + folder
		print("Running LexRank Summarizer for files in folder: ", folder)
		doc_summary = []
		summary = textRank.main(summary_length, path)
		for sentences in summary:
			text_append = re.sub("\n", "", sentences.getOGwords())
			text_append = text_append + " "
			doc_summary.append(text_append)
		total_summary.append(doc_summary)
	os.chdir("Data_DUC_2004/TextRank_results")

	for i in range(len(doc_folders)):
		myfile = doc_folders[i] + ".TextRank"
		f = open(myfile, 'w')
		for j in range(summary_length):
			f.write(total_summary[i][j])
		f.close()
