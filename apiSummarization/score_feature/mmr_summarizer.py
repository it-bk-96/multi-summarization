import math
import nltk
import os
from pyvi import ViTokenizer

porter = nltk.PorterStemmer()

root_abtract = os.getcwd()
nltk.download('punkt')


class sentence(object):

    def __init__(self, preproWords, position):
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.position = position

    def getPosition(self):
        return self.position

    def getPreProWords(self):
        return self.preproWords

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
            if stemmed_sentence :
                sentences.append(sentence(stemmed_sentence, [i, j]))
    return sentences


def processFileEnglish(documents):
    sentences = []

    for j in range(len(documents)):

        text_0 = documents[j]

        # tách câu
        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
        lines = sentence_token.tokenize(text_0.strip())

        # modelling each sentence in file as sentence object
        i = 0
        for line in lines:

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


def TFs(sentences):
    tfs = {}

    for sent in sentences:
        wordFreqs = sent.getWordFreq()
        for word in wordFreqs.keys():
            if tfs.get(word, 0) != 0:
                tfs[word] = tfs[word] + wordFreqs[word]
            else:
                tfs[word] = wordFreqs[word]
    return tfs


def IDFs(sentences):
    N = len(sentences)
    idfs = {}
    words = {}
    w2 = []
    for sent in sentences:
        for word in sent.getPreProWords():
            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0) + 1
    for word in words:
        n = words[word]

        # avoid zero division errors
        try:
            w2.append(n)
            idf = math.log10(float(N) / n)
        except ZeroDivisionError:
            idf = 0

        # reset variables
        idfs[word] = idf

    return idfs


def TF_IDF(sentences):
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}
    for word in tfs:
        tf_idfs = tfs[word] * idfs[word]

        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval


def sentenceSim(sentence1, sentence2, IDF_w):
    numerator = 0
    denominator = 0

    for word in sentence2.getPreProWords():
        numerator += sentence1.getWordFreq().get(word, 0) * sentence2.getWordFreq().get(word, 0) * IDF_w.get(word,
                                                                                                             0) ** 2

    for word in sentence1.getPreProWords():
        denominator += (sentence1.getWordFreq().get(word, 0) * IDF_w.get(word, 0)) ** 2

    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")


def buildQuery(sentences, TF_IDF_w, n):
    scores = list(TF_IDF_w.keys())
    scores.sort(reverse=True)

    i = 0
    j = 0
    queryWords = []

    while (i < n):
        words = TF_IDF_w[scores[j]]
        for word in words:
            queryWords.append(word)
            i = i + 1
            if (i > n):
                break
        j = j + 1

    # return the top selected words as a sentence
    return sentence(queryWords, [0, 0])


def bestSentence(sentences, query, IDF):
    best_sentence = None
    maxVal = float("-inf")

    for sent in sentences:
        similarity = sentenceSim(sent, query, IDF)

        if similarity > maxVal:
            best_sentence = sent
            maxVal = similarity
    sentences.remove(best_sentence)

    return best_sentence


def makeSummary(sentences, best_sentence, query, lambta, IDF):
    summary = [best_sentence]
    i = 0
    while sentences and i < 5:
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        i += 1
        sentences.remove(maxxer)

    return summary


def MMRScore(Si, query, Sj, lambta, IDF):
    Sim1 = sentenceSim(Si, query, IDF)
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentenceSim(Si, sent, IDF)
        value.append(Sim2)

    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr

    return MMR_SCORE


def getMMR(documents, language):
    if language == "vn":
        sentences = processFileVietNamese(documents)
    elif language == "en":
        sentences = processFileEnglish(documents)
    else:
        sentences = []
    IDF_w = IDFs(sentences)
    TF_IDF_w = TF_IDF(sentences)
    query = buildQuery(sentences, TF_IDF_w, 10)

    best1sentence = bestSentence(sentences, query, IDF_w)
    array_sentence = makeSummary(sentences, best1sentence, query, 0.5, IDF_w)
    summary = []
    for i in range(5):
        summary += [array_sentence[i].getPosition()]

    return summary
