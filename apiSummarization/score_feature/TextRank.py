import os
import numpy
import nltk
from pyvi import ViTokenizer

porter = nltk.PorterStemmer()

root_abtract = os.getcwd()
# root_abtract = os.path.dirname(os.getcwd())


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

    def normalize(self, numbers):
        max_number = max(numbers)
        normalized_numbers = []

        for number in numbers:
            normalized_numbers.append(number / max_number)

        return normalized_numbers

    def buildSummary(self, sentences, node_weights):

        summary = []
        top_index = [i for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)[:5]]

        for i in top_index:
            summary += [sentences[i].getPosition()]

        return summary

    def main(self, documents, language):
        if language == "vn":
            sentences = self.text.processFileVietNamese(documents)
        elif language == "en":
            sentences = self.text.processFileEnglish(documents)
        else:
            sentences = []
        num_nodes = len(sentences)
        graph = numpy.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # tinh toan độ trùng lặp giữa 2 sentences
                graph[i, j] = float(len(set(sentences[i].getStemmedWords()) & set(sentences[j].getStemmedWords()))) / (
                        len(sentences[i].getStemmedWords()) + len(sentences[j].getStemmedWords()))
                graph[j, i] = graph[i, j]

        node_weights = numpy.ones(num_nodes)
        self.PageRank(graph, node_weights)
        summary = self.buildSummary(sentences, node_weights)

        return summary


class sentence(object):

    def __init__(self, stemmedWords, position):
        self.stemmedWords = stemmedWords
        self.position = position

    def getStemmedWords(self):
        return self.stemmedWords

    def getPosition(self):
        return self.position


class Preprocessing(object):

    def processFileVietNamese(self, documents):
        # Đọc file
        stop_word = list(map(lambda x: "_".join(x.split()),
                             open(root_abtract + "/vietnamese-stopwords.txt",
                                  'r').read().split("\n")))

        sentences = []

        for j in range(len(documents)):

            text_0 = documents[j]

            # tách câu
            sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
            lines = sentence_token.tokenize(text_0.strip())

            # modelling each sentence in file as sentence object
            for i in range(len(lines)):
                line = lines[i]

                # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
                line = line.strip().lower()

                # tách từ
                stemmed_sentence = ViTokenizer.tokenize(line).split()
                stemmed_sentence = list(
                    filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                                     and x != '!' and x != '''"''' and x != "''" and x != '-' and x not in stop_word,
                           stemmed_sentence))
                if ((i + 1) == len(lines)) and (len(stemmed_sentence) <= 5):
                    break
                if stemmed_sentence:
                    sentences.append(sentence(stemmed_sentence, [i, j]))
        return sentences

    def processFileEnglish(self, documents):
        sentences = []

        for j in range(len(documents)):

            text_0 = documents[j]

            # tách câu
            sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
            lines = sentence_token.tokenize(text_0.strip())
            i = 0
            # modelling each sentence in file as sentence object
            for line in lines:
                # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
                line = line.strip().lower()
                line = nltk.word_tokenize(line)

                stemmed_sentence = [porter.stem(word) for word in line]
                stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'"
                                                         and x != '!' and x != '''"''' and x != "''" and x != "'s"
                                                         and x != '_' and x != '--' and x != "(" and x != ")" and x != ";",
                                               stemmed_sentence))

                if (len(stemmed_sentence) <= 4):
                    break
                if stemmed_sentence:
                    sentences.append(sentence(stemmed_sentence, [i, j]))
                    i += 1
        return sentences


def getTextRank(documents, language):
    textRank = TextRank()
    position = textRank.main(documents, language)

    return position


if __name__ == '__main__':
    path1 = root_abtract + '/Data/Documents/cluster_100/12180855.body.txt'
    path2 = root_abtract + '/Data/Documents/cluster_100/12180651.body.txt'
    path3 = root_abtract + '/Data/Documents/cluster_100/12180263.body.txt'
    path4 = root_abtract + '/Data/Documents/cluster_100/12180653.body.txt'
    text_1 = open(path1, 'r').read()
    text_2 = open(path2, 'r').read()
    text_3 = open(path3, 'r').read()
    text_4 = open(path4, 'r').read()
    print(getTextRank([text_1, text_2, text_3, text_4], "vn"))
