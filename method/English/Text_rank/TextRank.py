import os
import numpy
import nltk
import re

porter = nltk.PorterStemmer()
human_nu = 0
system_nu = 0


class TextRank(object):
    def __init__(self):
        self.text = Preprocessing()

    def PageRank(self, graph, node_weights, d=.85, iter=20):
        weight_sum = numpy.sum(graph, axis=0)
        while iter > 0:
            for i in range(len(node_weights)):
                temp = 0.0
                for j in range(len(node_weights)):
                    temp += graph[i, j] * node_weights[j] / weight_sum[j]
                node_weights[i] = 1 - d + (d * temp)
            iter -= 1

    def buildSummary(self, sentences, node_weights, n):

        top_index = [i for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)]
        summary = []
        sum_len = 0
        # keeping adding sentences until number of words exceeds summary length
        for i in top_index:
            if (sum_len > n):
                break
            summary += [sentences[i]]
            sum_len += len(sentences[i].getStemmedWords())

        global human_nu, system_nu
        human_nu += n
        system_nu += sum_len
        print(sum_len, n)

        return summary

    def main(self, n, path):
        sentences = self.text.openDirectory(path)
        num_nodes = len(sentences)
        graph = numpy.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # tinh toan độ trùng lặp giữa 2 sentences
                graph[i, j] = float(len(set(sentences[i].getStemmedWords()) & set(sentences[j].getStemmedWords()))) / (
                            len(sentences[i].getStemmedWords()) + len(sentences[j].getStemmedWords()))
                graph[j, i] = graph[i, j]

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
                    break

                if stemmed_sentence:
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
    root_directory = "/home/giangvu/Desktop/multi-summarization/"

    textRank = TextRank()
    doc_folders = os.listdir(root_directory + "Data/DUC_2007/Documents/")
    human_folder_path = root_directory + "Data/DUC_2007/Human_Summaries/"
    total_summary = []

    for folder in doc_folders:
        path = os.path.join(root_directory + "Data/DUC_2007/Documents/", '') + folder
        print("Running LexRank Summarizer for files in folder: ", folder)
        doc_summary = []

        file_human_1 = human_folder_path + "summary_" + folder[3:5] + ".A.1.txt"
        file_human_2 = human_folder_path + "summary_" + folder[3:5] + ".B.1.txt"
        file_human_3 = human_folder_path + "summary_" + folder[3:5] + ".C.1.txt"
        file_human_4 = human_folder_path + "summary_" + folder[3:5] + ".D.1.txt"
        text_1 = open(file_human_1, 'r').read()
        text_2 = open(file_human_2, 'r').read()
        text_3 = open(file_human_3, 'r').read()
        text_4 = open(file_human_4, 'r').read()
        summary_length = 0
        for el in [text_1, text_2, text_3, text_4]:
            llll = nltk.word_tokenize(el)

            # stemming words // đưa về từ gốc
            stemmedSent = [porter.stem(word) for word in llll]
            stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
                                                and x != '(' and x != ')' and x.find('&') == -1
                                                and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                and x != '``' and x != '--' and x != ':'
                                                and x != "''" and x != "'s", stemmedSent))
            summary_length += len(stemmedSent)
        summary_length = summary_length / 4

        summary = textRank.main(summary_length, path)
        for sentences in summary:
            text_append = re.sub("\n", "", sentences.getOGwords())
            text_append = text_append + " "
            doc_summary.append(text_append)
        results_folder = root_directory + "Data/DUC_2007/TextRank_results"
        with open(os.path.join(results_folder, (str(folder) + ".TextRank")), "w") as fileOut:
            fileOut.write("\n".join(doc_summary))

    print(system_nu, human_nu)
