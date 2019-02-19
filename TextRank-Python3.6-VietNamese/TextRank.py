import os
import numpy
import nltk
import re
from pyvi import ViTokenizer

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
		length_summary = len(ViTokenizer.tokenize(sentences[top_index[0]].getOGwords().strip()).split())

		for i in top_index:
			if (length_summary > n):
				break
			summary += [sentences[i]]
			length_summary += len(ViTokenizer.tokenize(sentences[i].getOGwords().strip()).split())

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

	def processFileVietNamese(self, file_path_and_name):
		try:
			# Đọc file
			f = open(file_path_and_name, 'r')
			text_0 = f.read()

			# tách câu
			sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
			lines = sentence_token.tokenize(text_0.strip())

			# setting the stemmer
			sentences = []

			# modelling each sentence in file as sentence object
			for i in range(len(lines)):
				line = lines[i]
				# giữ lại câu gốc
				OG_sent = line[:]

				# chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
				line = line.strip().lower()

				# tách từ
				stemmed_sentence = ViTokenizer.tokenize(line).split()
				stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
		                               and x != '!' and x != '''"''' and x != "''" and x != '-' and x not in stop_word,
				                               stemmed_sentence))
				if ((i + 1) == len(lines)) and (len(stemmed_sentence) <= 5):
					break
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
			sentences = sentences + self.processFileVietNamese(file_path)

		return sentences

if __name__ == '__main__':

	textRank = TextRank()
	doc_folders = os.listdir("Data_Chưa_tách_từ/Documents")
	total_summary = []
	human_folder_path = os.getcwd() + "/Data_Chưa_tách_từ/Human_Summaries/"

	stop_word = list(map(lambda x: "_".join(x.split()), open("/home/giangvu/Desktop/multi-summarization/vietnamese-stopwords.txt", 'r').read().split("\n")))

	for folder in doc_folders:
		path = os.path.join("Data_Chưa_tách_từ/Documents/", '') + folder
		print("Running TextRank Summarizer for files in folder: ", folder)
		doc_summary = []

		file_human_1 = human_folder_path + folder + ".ref1.txt"
		file_human_2 = human_folder_path + folder + ".ref2.txt"
		text_1 = open(file_human_1, 'r').read()
		text_2 = open(file_human_2, 'r').read()
		text_1_token = ViTokenizer.tokenize(text_1)
		text_2_token = ViTokenizer.tokenize(text_2)
		length_summary = int((len(text_1_token.split()) + len(text_1_token.split())) / 2)

		summary = textRank.main(length_summary, path)
		for sentences in summary:
			text_append = re.sub("\n", "", sentences.getOGwords())
			text_append = text_append + " "
			doc_summary.append(text_append)

		results_folder = os.getcwd() + "/Data_Chưa_tách_từ/TextRank_results"

		with open(os.path.join(results_folder, (str(folder) + ".TextRank")), "w") as fileOut:
			fileOut.write("\n".join(doc_summary))
