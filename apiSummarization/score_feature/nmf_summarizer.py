import os
import nltk
import numpy as np
from pyvi import ViTokenizer
from sklearn.decomposition import NMF

porter = nltk.PorterStemmer()

root_abtract = os.getcwd()
# root_abtract = os.path.dirname(os.getcwd())

class sentence(object):

    def __init__(self, stemmedWords, position):

        self.stemmedWords = stemmedWords
        self.position = position
        self.wordFrequencies = self.sentenceWordFreqs()
        self.lexRankScore = None

    def getLexRankScore(self):
        return self.LexRankScore

    def setLexRankScore(self, score):
        self.LexRankScore = score

    def getStemmedWords(self):
        return self.stemmedWords

    def getWordFreqs(self):
        return self.wordFrequencies

    def getPosition(self):
        return self.position

    def sentenceWordFreqs(self):
        wordFreqs = {}
        for word in self.stemmedWords:
            if word not in wordFreqs.keys():
                wordFreqs[word] = 1
            else:
                wordFreqs[word] = wordFreqs[word] + 1

        return wordFreqs


def processFileVietNamese(documents):
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


def processFileEnglish(documents):
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


def normalize(numbers):
    max_number = max(numbers)
    normalized_numbers = []

    for number in numbers:
        normalized_numbers.append(number / max_number)

    return normalized_numbers


def getNMF(documents, language):
    if language == "vn":
        sentences = processFileVietNamese(documents)
    elif language == "en":
        sentences = processFileEnglish(documents)
    else:
        sentences = []  # tf
    vocabulary = []
    for sent in sentences:
        vocabulary = vocabulary + sent.getStemmedWords()
    vocabulary = list(set(vocabulary))
    A = np.zeros(shape=(len(vocabulary), len(sentences)))
    for i in range(len(sentences)):
        tf_sentence = sentences[i].getWordFreqs()
        for word in tf_sentence.keys():
            index = vocabulary.index(word)
            A[index][i] += tf_sentence[word]

    rank_A = np.linalg.matrix_rank(A)
    model = NMF(n_components=rank_A, init='random', random_state=0)
    W = model.fit_transform(A)
    H = model.components_
    scores = np.sum(H, axis=0)

    summary = []
    top_index = [i for i, j in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]]

    for i in top_index:
        summary += [sentences[i].getPosition()]

    return summary


if __name__ == '__main__':
    path1 = root_abtract + '/Data/Documents/cluster_1/12240106.body.txt'
    path2 = root_abtract + '/Data/Documents/cluster_1/12240586.body.txt'
    path3 = root_abtract + '/Data/Documents/cluster_1/12241528.body.txt'
    text_1 = open(path1, 'r').read()
    text_2 = open(path2, 'r').read()
    text_3 = open(path3, 'r').read()
    print(getNMF([text_1, text_2, text_3], "vn"))
