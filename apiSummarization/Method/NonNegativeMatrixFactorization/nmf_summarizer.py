import os
import nltk
from pyvi import ViTokenizer
import numpy as np
import pickle
from sklearn.decomposition import NMF
root_abtract = os.getcwd()

with open(root_abtract + "/models/w2v.model", 'rb') as f:
    model_w2v = pickle.load(f)


class sentence(object):

    # ------------------------------------------------------------------------------
    # Description	: Constructor to initialize the setence object
    # Parameters  	: docName, name of the document/file
    #				  preproWords, words of the file after the stemming process
    #				  originalWords, actual words before stemming
    # Return 		: None
    # ------------------------------------------------------------------------------
    def __init__(self, preproWords, originalWords):
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.originalWords = originalWords
        self.score = 0
        self.vectorSentence = self.caculateVector(preproWords)

    def caculateVector(self, preproWords):
        result = np.zeros(300)
        for word in preproWords:
            try:
                result += np.array(model_w2v[word])
            except:
                continue
        return result

    def getVectorSentence(self):
        return self.vectorSentence

    def setScore(self, score):
        self.score = score

    def getScore(self):
        return self.score

    def getPreProWords(self):
        return self.preproWords

    def getOriginalWords(self):
        return self.originalWords

    def getWordFreq(self):
        return self.wordFrequencies

    def sentenceWordFreq(self):
        wordFreq = {}
        for word in self.preproWords:
            if word not in wordFreq.keys():
                wordFreq[word] = 1
            else:
                wordFreq[word] += + 1

        return wordFreq


def processFile(documents):
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

    return [sentences[index] for index in array_index]


def getNMF_W2V(documents, length):
    sentences = processFile(documents)

    A = np.zeros(shape=(300, len(sentences)))
    for i in range(len(sentences)):
        vector = sentences[i].getVectorSentence()
        for j in range(300):
            if vector[j] < 0:
                A[j][i] = 0
            else:
                A[j][i] = vector[j]

    rank_A = np.linalg.matrix_rank(A)

    model = NMF(n_components=rank_A, init='random', random_state=0)
    W = model.fit_transform(A)
    H = model.components_

    # build summary
    summary = makeSummary_Gong_Liu(sentences, length, H)

    final_summary = ""
    for sent in summary:
        final_summary = final_summary + sent.getOriginalWords() + "\n"
    final_summary = final_summary[:-1]

    return final_summary


if __name__ == '__main__':
    path1 = root_abtract + '/Data/Documents/cluster_1/12240106.body.txt'
    path2 = root_abtract + '/Data/Documents/cluster_1/12240586.body.txt'
    path3 = root_abtract + '/Data/Documents/cluster_1/12241528.body.txt'
    text_1 = open(path1, 'r').read()
    text_2 = open(path2, 'r').read()
    text_3 = open(path3, 'r').read()
    print(getNMF_W2V([text_1, text_2, text_3], 100))
