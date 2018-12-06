import os
import math
import numpy

numpy.set_printoptions(threshold=numpy.nan)
import copy
import nltk
from bs4 import BeautifulSoup
import re


class LexRank(object):
    def __init__(self):
        self.text = Preprocessing()
        self.sim = DocumentSim()

    def score(self, sentences, idfs, CM, t):

        Degree = [0 for i in sentences]
        L = [0 for i in sentences]
        n = len(sentences)

        for i in range(n):
            for j in range(n):
                CM[i][j] = self.sim.sim(sentences[i], sentences[j], idfs)

                if CM[i][j] > t:
                    CM[i][j] = 1
                    Degree[i] += 1

                else:
                    CM[i][j] = 0

        for i in range(n):
            for j in range(n):
                CM[i][j] = CM[i][j] / float(Degree[i])

        L = self.PowerMethod(CM, n, 0.2)
        normalizedL = self.normalize(L)

        for i in range(len(normalizedL)):
            score = normalizedL[i]
            sentence = sentences[i]
            sentence.setLexRankScore(score)

        return sentences

    def PowerMethod(self, CM, n, e):
        Po = numpy.array([1 / float(n) for i in range(n)])
        t = 0
        delta = float('-inf')
        M = numpy.array(CM)

        while delta < e:
            t = t + 1
            M = M.transpose()
            P1 = numpy.dot(M, Po)
            diff = numpy.subtract(P1, Po)

            # norm 2 cua matrix
            delta = numpy.linalg.norm(diff)
            Po = numpy.copy(P1)

        return list(Po)

    def buildMatrix(self, sentences):
        # CM = []
        # for s in sentences:
        #     tmp = []
        #     for s in sentences:
        #         tmp.append(0)
        #     CM.append(0)

        # build our matrix
        CM = [[0 for s in sentences] for s in sentences]

        # print numpy.array(CM)
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                CM[i][j] = 0

        # print 'vl'
        # print numpy.array(CM)
        return CM

    def buildSummary(self, sentences, n):
        sentences = sorted(sentences, key=lambda x: x.getLexRankScore(), reverse=True)
        summary = []
        # sum_len = 0

        # while sum_len < n:
        #     summary += [sentences[i]]
        #     sum_len += len(sentences[i].getStemmedWords())

        for i in range(n):
            summary += [sentences[i]]
        return summary

    def normalize(self, numbers):
        max_number = max(numbers)
        normalized_numbers = []

        for number in numbers:
            normalized_numbers.append(number / max_number)

        return normalized_numbers

    def main(self, n, path):
        sentences = self.text.openDirectory(path)
        idfs = self.sim.IDFs(sentences)
        CM = self.buildMatrix(sentences)

        sentences = self.score(sentences, idfs, CM, 0.1)

        summary = self.buildSummary(sentences, n)

        return summary


class sentence(object):
    def __init__(self, docName, stemmedWords, OGwords):

        self.stemmedWords = stemmedWords
        self.docName = docName
        self.OGwords = OGwords
        self.wordFrequencies = self.sentenceWordFreqs()
        self.lexRankScore = None

    def getStemmedWords(self):
        return self.stemmedWords

    def getDocName(self):
        return self.docName

    def getOGwords(self):
        return self.OGwords

    def getWordFreqs(self):
        return self.wordFrequencies

    def getLexRankScore(self):
        return self.LexRankScore

    def setLexRankScore(self, score):
        self.LexRankScore = score

    def sentenceWordFreqs(self):
        wordFreqs = {}
        for word in self.stemmedWords:
            if word not in wordFreqs.keys():
                wordFreqs[word] = 1
            else:
                wordFreqs[word] = wordFreqs[word] + 1

        return wordFreqs


class Preprocessing(object):
    def processFile(self, file_path_and_name):
        try:
            f = open(file_path_and_name, 'r')
            text = f.read()

            # soup = BeautifulSoup(text,"html.parser")
            # text = soup.getText()
            # text = re.sub("APW19981212.0848","",text)
            # text = re.sub("APW19981129.0668","",text)
            # text = re.sub("NEWSWIRE","",text)
            # text_1 = re.search(r"<TEXT>.*</TEXT>",text, re.DOTALL)
            # text_1 = re.sub("<TEXT>\n","",text_1.group(0))
            # text_1 = re.sub("\n</TEXT>","",text_1)

            # replace all types of quotations by normal quotes
            text_1 = re.sub("\n", " ", text)
            text_1 = re.sub(" +", " ", text_1)
            # text_1 = re.sub("\'\'","\"",text_1)
            # text_1 = re.sub("\`\`","\"",text_1)


            sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

            lines = sent_tokenizer.tokenize(text_1.strip())
            text_1 = lines

            sentences = []
            porter = nltk.PorterStemmer()

            for sent in lines:
                OG_sent = sent[:]
                sent = sent.strip().lower()
                line = nltk.word_tokenize(sent)

                stemmed_sentence = [porter.stem(word) for word in line]
                stemmed_sentence = filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'"
                                                    and x != '!' and x != '''"''' and x != "''" and x != "'s",
                                          stemmed_sentence)
                if stemmed_sentence != []:
                    sentences.append(sentence(file_path_and_name, stemmed_sentence, OG_sent))

            return sentences


        except IOError:
            print 'Oops! File not found', file_path_and_name
            return [sentence(file_path_and_name, [], [])]

    def get_all_files(self, path=None):
        retval = []

        if path == None:
            path = os.getcwd()

        for root, dirs, files in os.walk(path):
            for name in files:
                if name == '.gitkeep':
                    continue
                retval.append(os.path.join(root, name))
        return retval

    def openDirectory(self, path=None):
        file_paths = self.get_all_files(path)
        sentences = []
        for file_path in file_paths:
            sentences = sentences + self.processFile(file_path)

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
        idf = 0
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

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

if __name__ == '__main__':

    lexRank = LexRank()
    doc_folders = os.walk(DIR_PATH + '/data').next()[1]
    print doc_folders
    total_summary = []
    summary_length = 20
    for i in range(len(doc_folders)):
        path = os.path.join(DIR_PATH + '/data', '') + doc_folders[i]
        doc_summary = []
        summary = []
        print path
        print '-' * 20
        summary = lexRank.main(summary_length, path)
        # print i
        for sentences in summary:
            # print "\n", sentences.getOGwords(), "\n"
            text_append = re.sub("\n", "", sentences.getOGwords())
            # text_append = text_append.strip("'")
            text_append = text_append + " "
            doc_summary.append(text_append)
        total_summary.append(doc_summary)
        # print total_summary
        # break

    os.chdir(DIR_PATH + "/summaries/" + "LexRank_results")
    text_len = 100
    tmp = 'summary_'
    for i in range(len(doc_folders)):
        myfile = tmp + doc_folders[i] + ".1.txt"
        f = open(myfile, 'w')

        tmp_len = 0
        for j in range(summary_length):
            # print ('------------------------------------')
            f.write(total_summary[i][j])
            tmp_len += (total_summary[i][j].count(' '))
            # print total_summary[i][j]
            # print tmp_len
            if tmp_len > 100:
                break
        f.close()