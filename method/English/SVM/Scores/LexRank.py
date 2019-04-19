import math
import numpy

class LexRank(object):
    def __init__(self):
        self.text = Preprocessing()
        self.sim = DocumentSim()

    def score(self, sentences, idfs, CM):

        Degree = [0 for i in sentences]
        n = len(sentences)

        for i in range(n):
            for j in range(n):
                # tf.idf giữa 2 câu
                CM[i][j] = self.sim.sim(sentences[i], sentences[j], idfs)
                Degree[i] += CM[i][j]

        for i in range(n):
            for j in range(n):
                CM[i][j] = CM[i][j] / float(Degree[i])

        L = self.PageRank(CM, n)
        normalizedL = self.normalize(L)

        return [normalizedL[i] for i in range(len(normalizedL))]

    def PageRank(self, CM, n, maxerr=.0001):
        Po = numpy.zeros(n)
        P1 = numpy.ones(n)
        M = numpy.array(CM)
        t = 0
        while (numpy.sum(numpy.abs(P1 - Po)) > maxerr) and (t < 500):
            Po = numpy.copy(P1)
            t = t + 1
            P1 = numpy.matmul(Po, M)

        return list(Po)

    def buildMatrix(self, sentences):

        # build our matrix
        CM = [[0 for s in sentences] for s in sentences]
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                CM[i][j] = 0
        return CM

    def normalize(self, numbers):
        max_number = max(numbers)
        normalized_numbers = []

        for number in numbers:
            normalized_numbers.append(number / max_number)

        return normalized_numbers

    def main(self, documents):
        sentences = self.text.processFileVietNamese(documents)
        idfs = self.sim.IDFs(sentences)
        CM = self.buildMatrix(sentences)

        node_weights = self.score(sentences, idfs, CM)

        return node_weights


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


class Preprocessing(object):

    def processFileVietNamese(self, documents):

        sentences = []
        for sent in documents:
            sentences.append(sentence(sent))

        return sentences
    # def processFileVietNamese(self, documents):
    #     # Đọc file
    #     stop_word = list(map(lambda x: "_".join(x.split()),
    #                          open("/home/giangvu/Desktop/multi-summarization/vietnamese-stopwords.txt",
    #                               'r').read().split("\n")))
    #
    #     sentences = []
    #     for document in documents:
    #
    #         text_0 = document
    #
    #         # tách câu
    #         sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    #         lines = sentence_token.tokenize(text_0.strip())
    #
    #         # modelling each sentence in file as sentence object
    #         for i in range(len(lines)):
    #             line = lines[i]
    #             # giữ lại câu gốc
    #             OG_sent = line[:]
    #
    #             # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
    #             line = line.strip().lower()
    #
    #             # tách từ
    #             stemmed_sentence = ViTokenizer.tokenize(line).split()
    #             stemmed_sentence = list(
    #                 filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
    #                                  and x != '!' and x != '''"''' and x != "''" and x != '-' and x not in stop_word,
    #                        stemmed_sentence))
    #             if ((i + 1) == len(lines)) and (len(stemmed_sentence) <= 5):
    #                 break
    #             if stemmed_sentence != []:
    #                 sentences.append(sentence(stemmed_sentence))
    #     return sentences

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


def getLexRank(documents):
    lexRank = LexRank()
    scores = lexRank.main(documents)

    return scores
#
# path1 = '/home/giangvu/Desktop/api-summarization/Data/Documents/cluster_1/12240106.body.txt'
# path2 = '/home/giangvu/Desktop/api-summarization/Data/Documents/cluster_1/12240586.body.txt'
# path3 = '/home/giangvu/Desktop/api-summarization/Data/Documents/cluster_1/12241528.body.txt'
# text_1 = open(path1, 'r').read()
# text_2 = open(path2, 'r').read()
# text_3 = open(path3, 'r').read()
# print(getLexRank([text_1, text_2, text_3]))
# print(len(getLexRank([text_1, text_2, text_3])))