import os
import numpy
import nltk
import re
from pyvi import ViTokenizer
root_abtract = os.getcwd()


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

        summary = []
        top_index = [i for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)[:n]]
        length_summary = len(ViTokenizer.tokenize(sentences[top_index[0]].getOGwords().strip()).split())

        for i in top_index:
            if (length_summary > n):
                break
            summary += [sentences[i]]
            length_summary += len(ViTokenizer.tokenize(sentences[i].getOGwords().strip()).split())

        return summary

    def main(self, n, documents):
        sentences = self.text.processFileVietNamese(documents)
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

    def processFileVietNamese(self, documents):
        # Đọc file
        stop_word = list(map(lambda x: "_".join(x.split()),
                             open(root_abtract + "/vietnamese-stopwords.txt",
                                  'r').read().split("\n")))

        sentences = []
        for document in documents:

            text_0 = document

            # tách câu
            sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
            lines = sentence_token.tokenize(text_0.strip())

            # modelling each sentence in file as sentence object
            for i in range(len(lines)):
                line = lines[i]
                # giữ lại câu gốc
                OG_sent = line[:]

                # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
                line = line.strip().lower()

                # tách từ
                stemmed_sentence = ViTokenizer.tokenize(line).split()
                stemmed_sentence = list(
                    filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                                     and x != '!' and x != '''"''' and x != "''" and x != '-' and x != '>>' and x not in stop_word,
                           stemmed_sentence))
                if ((i + 1) == len(lines)) and (len(stemmed_sentence) <= 5):
                    break
                if stemmed_sentence != []:
                    sentences.append(sentence(stemmed_sentence, OG_sent))
        return sentences


def getTextRank(documents, length_summary):
    textRank = TextRank()
    doc_summary = []

    summary = textRank.main(length_summary, documents)
    for sentences in summary:
        text_append = re.sub("\n", "", sentences.getOGwords())
        text_append = text_append + " "
        doc_summary.append(text_append)

    return "\n".join(doc_summary)


if __name__ == '__main__':
    path1 = root_abtract + '/Data/Documents/cluster_1/12240106.body.txt'
    path2 = root_abtract + '/Data/Documents/cluster_1/12240586.body.txt'
    path3 = root_abtract + '/Data/Documents/cluster_1/12241528.body.txt'
    text_1 = open(path1, 'r').read()
    text_2 = open(path2, 'r').read()
    text_3 = open(path3, 'r').read()
    print(getTextRank([text_1, text_2, text_3], 100))
