from __future__ import print_function
import os
import math
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pyvi import ViTokenizer
porter = nltk.PorterStemmer()

root_abtract = os.getcwd()
nltk.download('punkt')


class sentence(object):

    def __init__(self, preproWords, weightedPosition, position):
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.weightedPosition = weightedPosition
        self.position = position

    def getPosition(self):
        return self.position

    def getPreProWords(self):
        return self.preproWords

    def getWeightedPosition(self):
        return self.weightedPosition

    def getWordFreq(self):
        return self.wordFrequencies

    def sentenceWordFreq(self):
        wordFreq = {}
        for word in self.preproWords:
            if word not in wordFreq.keys():
                wordFreq[word] = 1
            else:
                wordFreq[word] = wordFreq[word] + 1
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

                sentences.append(sentence(stemmed_sentence, float(1 / (i + 1)), [i, j]))
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



# ---------------------------------------------------------------------------------
# Description	: Function to find the term frequencies of the words in the
#				  sentences present in the provided document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, term frequency score
# ---------------------------------------------------------------------------------
def TFs(sentences):
    # initialize tfs dictonary
    tfs = {}

    # for every sentence in document cluster
    for sent in sentences:
        # retrieve word frequencies from sentence object
        wordFreqs = sent.getWordFreq()

        # for every word
        for word in wordFreqs.keys():
            # if word already present in the dictonary
            if tfs.get(word, 0) != 0:
                tfs[word] = tfs[word] + wordFreqs[word]
            # else if word is being added for the first time
            else:
                tfs[word] = wordFreqs[word]
    return tfs


# ---------------------------------------------------------------------------------
# Description	: Function to find the inverse document frequencies of the words in
#				  the sentences present in the provided document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, inverse document frequency score
# ---------------------------------------------------------------------------------
def IDFs(sentences):
    N = len(sentences)
    idfs = {}
    words = {}
    w2 = []
    # every sentence in our cluster
    for sent in sentences:

        # every word in a sentence
        for word in sent.getPreProWords():
            # not to calculate a word's IDF value more than once
            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0) + 1

    # for each word in words
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


# ---------------------------------------------------------------------------------
# Description	: Function to find TF-IDF score of the words in the document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, TF-IDF score
# ---------------------------------------------------------------------------------
def TF_IDF(sentences):
    # method variables
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}

    # for every word
    for word in tfs:
        # calculate every word's tf-idf score
        tf_idfs = tfs[word] * idfs[word]

        # add word and its tf-idf score to dictionary
        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)
    return retval


# ---------------------------------------------------------------------------------
# Description	: Function to find the sentence similarity for a pair of sentences
#				  by calculating cosine similarity
# Parameters	: sentence1, first sentence
#				  sentence2, second sentence to which first sentence has to be compared
#				  IDF_w, dictinoary of IDF scores of words in the document cluster
# Return 		: cosine similarity score
# ---------------------------------------------------------------------------------
def sentenceSim(sentence1, sentence2, IDF_w):
    numerator = 0
    denominator = 0

    for word in sentence2.getPreProWords():
        numerator += sentence1.getWordFreq().get(word, 0) * sentence2.getWordFreq().get(word, 0) * IDF_w.get(word,
                                                                                                             0) ** 2

    for word in sentence1.getPreProWords():
        denominator += (sentence1.getWordFreq().get(word, 0) * IDF_w.get(word, 0)) ** 2

    # check for divide by zero cases and return back minimal similarity
    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")


# ---------------------------------------------------------------------------------
# Description	: Function to build a query of n words on the basis of TF-IDF value
# Parameters	: sentences, sentences of the document cluster
#				  IDF_w, IDF values of the words
#				  n, desired length of query (number of words in query)
# Return 		: query sentence consisting of best n words
# ---------------------------------------------------------------------------------
def buildQuery(sentences, TF_IDF_w, n):
    # sort in descending order of TF-IDF values
    scores = list(TF_IDF_w.keys())
    scores.sort(reverse=True)

    i = 0
    j = 0
    queryWords = []

    # select top n words
    while (i < n):
        words = TF_IDF_w[scores[j]]
        for word in words:
            queryWords.append(word)
            i = i + 1
            if (i > n):
                break
        j = j + 1

    # return the top selected words as a sentence
    return sentence(queryWords, queryWords, 0)


# ---------------------------------------------------------------------------------
# Description	: Function to find the best sentence in reference to the query
# Parameters	: sentences, sentences of the document cluster
#				  query, reference query
#				  IDF, IDF value of words of the document cluster
# Return 		: best sentence among the sentences in the document cluster
# ---------------------------------------------------------------------------------
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


def PageRank(graph, node_weights, d=.85, iter=20):
    weight_sum = np.sum(graph, axis=0)
    while iter > 0:
        for i in range(len(node_weights)):
            temp = 0.0
            for j in range(len(node_weights)):
                temp += graph[i, j] * node_weights[j] / weight_sum[j]
            node_weights[i] = 1 - d + (d * temp)
        iter -= 1

def normalize(numbers):
    max_number = max(numbers)
    normalized_numbers = []

    for number in numbers:
        normalized_numbers.append(number / max_number)

    return normalized_numbers

def sim_cosin(sentence1, sentence2):

    numerator = 0
    denom1 = 0
    denom2 = 0

    for i in range(len(sentence1)):
        numerator += sentence1[i] * sentence2[i]

    for i in range(len(sentence1)):
        denom2 += sentence1[i] ** 2

    for i in range(len(sentence2)):
        denom1 += sentence2[i] ** 2

    try:
        return numerator / (math.sqrt(denom1) * math.sqrt(denom2))

    except ZeroDivisionError:
        return float("-inf")


def makeSummaryPositionMMR(sentences, query, k_cluster, summary_length, lambta, IDF):
    # k mean
    # create vocabulary
    vocabulary = []
    for sent in sentences:
        vocabulary = vocabulary + sent.getPreProWords()
    vocabulary = list(set(vocabulary))

    # clustering by tf-idf from vocabulary
    A = np.zeros(shape=(len(sentences), len(vocabulary)))
    for i in range(len(sentences)):
        for word in sentences[i].getWordFreq():
            index = vocabulary.index(word)
            A[i][index] = sentences[i].getWordFreq().get(word, 0) ** IDF[word]
    kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(A)

    # get k sentence nearest k cluster
    avg = []
    for j in range(k_cluster):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, A)
    ordering = sorted(range(k_cluster), key=lambda k: avg[k])

    k_mean_sentences = [sentences[closest[idx]] for idx in ordering]

    # best_sentence = bestSentence(k_mean_sentences, query, IDF)
    summary = []
    position = [sen.getWeightedPosition() for sen in k_mean_sentences]

    # keeping adding sentences until number of words exceeds summary length
    test = 1
    while test < 6 and k_mean_sentences:
        max_value = max(position)
        if position.count(max_value) == 1:
            maxxer = max(k_mean_sentences, key=lambda item: item.getWeightedPosition())
            summary.append(maxxer)
            k_mean_sentences.remove(maxxer)
            position = [sen.getWeightedPosition() for sen in k_mean_sentences]
            test += 1
        else:
            MMRval = {}
            list_p = []
            for i in range(len(position)):
                if position[i] == max_value:
                    list_p.append(i)
            for i in list_p:
                MMRval[i] = MMRScore(k_mean_sentences[i], query, summary, lambta, IDF)
            last_p = max(MMRval, key=MMRval.get)
            maxxer = k_mean_sentences[last_p]
            summary.append(maxxer)
            k_mean_sentences.remove(maxxer)
            position = [sen.getWeightedPosition() for sen in k_mean_sentences]
            test += 1

    return summary

# ---------------------------------------------------------------------------------
# Description	: Function to calculate the MMR score given a sentence, the query
#				  and the current best set of sentences
# Parameters	: Si, particular sentence for which the MMR score has to be calculated
#				  query, query sentence for the particualr document cluster
#				  Sj, the best sentences that are already selected
#				  lambta, lambda value in the MMR formula
#				  IDF, IDF value for words in the cluster
# Return 		: name
# ---------------------------------------------------------------------------------
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


def getK_cluster(sentences, n):
    min_word = 1000
    mean_word = 0
    for sentence in sentences:
        if len(sentence.getPreProWords()) < min_word:
            min_word = len(sentence.getPreProWords())
        mean_word += len(sentence.getPreProWords())
    # result = n // (min_word) + 1
    result = mean_word // len(sentences) + 1
    return result if result <= len(sentences) else len(sentences) // 2


def getKmeanPMMR(documents, language, length_summary):
    if language == "vn":
        sentences = processFileVietNamese(documents)
    else:
        sentences = []
    IDF_w = IDFs(sentences)
    TF_IDF_w = TF_IDF(sentences)
    query = buildQuery(sentences, TF_IDF_w, 10)
    k_cluster = getK_cluster(sentences, length_summary)

    array_sentence = makeSummaryPositionMMR(sentences, query, k_cluster, length_summary, 0.5, IDF_w)
    summary = []
    for i in range(5):
        summary += [array_sentence[i].getPosition()]

    return summary