import os
import math
import numpy
import nltk
from pyvi import ViTokenizer

porter = nltk.PorterStemmer()

root_abtract = os.getcwd()


# root_abtract = os.path.dirname(os.getcwd())


class LexRank(object):
    def __init__(self):
        self.text = Preprocessing()
        self.sim = DocumentSim()

    def score(self, sentences, idfs, CM, t):

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

        for i in range(len(normalizedL)):
            score = normalizedL[i]
            sentence = sentences[i]
            sentence.setLexRankScore(score)

        return sentences

    def PageRank(self, CM, n, maxerr=.0001):
        Po = numpy.zeros(n)
        P1 = numpy.ones(n)
        M = numpy.array(CM)
        t = 0
        while (numpy.sum(numpy.abs(P1 - Po)) > maxerr) and (t < 500):
            Po = numpy.copy(P1)
            t = t + 1
            P1 = numpy.matmul(Po, M)
        #     print(numpy.sum(numpy.abs(P1 - Po)))
        # print(t)
        return list(Po)

    def buildMatrix(self, sentences):

        # build our matrix
        CM = [[0 for s in sentences] for s in sentences]
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                CM[i][j] = 0
        return CM

    def buildSummary(self, sentences):
        sentences = sorted(sentences, key=lambda x: x.getLexRankScore(), reverse=True)
        summary = []
        for i in range(5):
            summary += [sentences[i].getPosition()]

        return summary

    def normalize(self, numbers):
        max_number = max(numbers)
        normalized_numbers = []

        for number in numbers:
            normalized_numbers.append(number / max_number)

        return normalized_numbers

    def main(self, documents, language):
        if language == "vn":
            sentences = self.text.processFileVietNamese(documents)
        elif language == "en":
            sentences = self.text.processFileEnglish(documents)
        else:
            sentences = []
        idfs = self.sim.IDFs(sentences)
        CM = self.buildMatrix(sentences)

        sentences = self.score(sentences, idfs, CM, 0.1)

        summary = self.buildSummary(sentences)

        return summary


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


def getLexRank(documents, language):
    lexRank = LexRank()
    position = lexRank.main(documents, language)

    return position
