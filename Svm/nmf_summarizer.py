import nltk
import numpy as np
from pyvi import ViTokenizer
from sklearn.decomposition import NMF


class sentence(object):

    def __init__(self, stemmedWords):

        self.stemmedWords = stemmedWords
        self.wordFrequencies = self.sentenceWordFreqs()

    def getStemmedWords(self):
        return self.stemmedWords

    def getWordFreqs(self):
        return self.wordFrequencies

    def sentenceWordFreqs(self):
        wordFreqs = {}
        for word in self.stemmedWords:
            if word not in wordFreqs.keys():
                wordFreqs[word] = 1
            else:
                wordFreqs[word] = wordFreqs[word] + 1

        return wordFreqs


def processFileVietNamese(documents):

    sentences = []

    for sent in documents:
        sentences.append(sentence(sent))

    return sentences
# def processFileVietNamese(documents):
#         # Đọc file
#         stop_word = list(map(lambda x: "_".join(x.split()),
#                              open("/home/giangvu/Desktop/multi-summarization/vietnamese-stopwords.txt",
#                                   'r').read().split("\n")))
#
#         sentences = []
#         for document in documents:
#
#             text_0 = document
#
#             # tách câu
#             sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
#             lines = sentence_token.tokenize(text_0.strip())
#
#             # modelling each sentence in file as sentence object
#             for i in range(len(lines)):
#                 line = lines[i]
#                 # giữ lại câu gốc
#                 OG_sent = line[:]
#
#                 # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
#                 line = line.strip().lower()
#
#                 # tách từ
#                 stemmed_sentence = ViTokenizer.tokenize(line).split()
#                 stemmed_sentence = list(
#                     filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
#                                      and x != '!' and x != '''"''' and x != "''" and x != '-' and x not in stop_word,
#                            stemmed_sentence))
#                 if ((i + 1) == len(lines)) and (len(stemmed_sentence) <= 5):
#                     break
#                 if stemmed_sentence != []:
#                     sentences.append(sentence(stemmed_sentence))
#         return sentences



def normalize(numbers):
    max_number = max(numbers)
    normalized_numbers = []

    for number in numbers:
        normalized_numbers.append(number / max_number)

    return normalized_numbers


def getNMF(documents):
    sentences = processFileVietNamese(documents)
    # tf
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
    return normalize(list(scores))

#
# path1 = '/home/giangvu/Desktop/api-summarization/Data/Documents/cluster_1/12240106.body.txt'
# path2 = '/home/giangvu/Desktop/api-summarization/Data/Documents/cluster_1/12240586.body.txt'
# path3 = '/home/giangvu/Desktop/api-summarization/Data/Documents/cluster_1/12241528.body.txt'
# text_1 = open(path1, 'r').read()
# text_2 = open(path2, 'r').read()
# text_3 = open(path3, 'r').read()
# print(getNMF([text_1, text_2, text_3]))
