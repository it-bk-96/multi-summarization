from __future__ import print_function
import os
import re
import math
import nltk
import numpy as np
porter = nltk.PorterStemmer()
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

root_directory = "/home/giangvu/Desktop/multi-summarization/"

human_nu = 0
system_nu = 0


class sentence(object):

    def __init__(self, docName, preproWords, originalWords, weightedPosition):
        self.docName = docName
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.originalWords = originalWords
        self.weightedPosition = weightedPosition
        self.weightedPageRank = 0

    def getWeightedPosition(self):
        return self.weightedPosition

    def getWeightedPageRank(self):
        return self.weightedPageRank

    def setWeightedPageRank(self, value):
        self.weightedPageRank = value

    def getDocName(self):
        return self.docName

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
                wordFreq[word] = wordFreq[word] + 1
        return wordFreq


def processFile(file_path_and_name):

    try:
        f = open(file_path_and_name, 'r')
        text_0 = f.read()

        # code 2007
        text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
        text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
        text_1 = re.sub("\n</TEXT>", "", text_1)

        text_1 = re.sub("<P>", "", text_1)
        text_1 = re.sub("</P>", "", text_1)
        text_1 = re.sub("\n", " ", text_1)
        text_1 = re.sub("\"", "\"", text_1)
        text_1 = re.sub("''", "\"", text_1)
        text_1 = re.sub("``", "\"", text_1)
        text_1 = re.sub(" +", " ", text_1)
        text_1 = re.sub(" _ ", "", text_1)

        text_1 = re.sub(r"\(AP\) _", " ", text_1)
        text_1 = re.sub("&\w+;", " ", text_1)

        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        lines = sent_tokenizer.tokenize(text_1.strip())
        # setting the stemmer

        # preprocess line[0]

        index = lines[0].find("--")
        if index != -1:
            lines[0] = lines[0][index + 2:]
        index = lines[0].find(" _ ")
        if index != -1:
            lines[0] = lines[0][index + 3:]

        sentences = []

        for i in range(len(lines)):
            sent = lines[i].strip()
            OG_sent = sent[:]
            sent = sent.lower()
            line = nltk.word_tokenize(sent)

            stemmed_sentence = [porter.stem(word) for word in line]
            stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
                                                and x != '(' and x != ')' and x.find('&') == -1
                                                and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                and x != '``' and x != '--' and x != ':'
                                                and x != "''" and x != "'s", stemmed_sentence))

            if (len(stemmed_sentence) <= 4):
                break
            if stemmed_sentence:
                if i == (len(lines) - 1):
                    weight_position = float(1.0)
                else:
                    weight_position = float(1 / (i + 1))
                sentences.append(sentence(file_path_and_name, stemmed_sentence, OG_sent, weight_position))

        return sentences

    except IOError:
        print('Oops! File not found', file_path_and_name)
        return [sentence(file_path_and_name, [], [], 0)]


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
    return sentence("query", queryWords, queryWords, 0)


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

def PageRank(graph, node_weights, d=.85, iter=20):
    weight_sum = np.sum(graph, axis=0)
    while iter > 0:
        for i in range(len(node_weights)):
            temp = 0.0
            for j in range(len(node_weights)):
                temp += graph[i, j] * node_weights[j] / weight_sum[j]
            node_weights[i] = 1 - d + (d * temp)
        iter -= 1

def makeSummaryPageRankPMMR(sentences, query, k_cluster, summary_length, lambta, IDF):
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
    vector_clusters = kmeans.cluster_centers_

    num_nodes = len(vector_clusters)
    graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # tinh toan độ trùng lặp giữa 2 sentences
            graph[i, j] = float(sim_cosin(vector_clusters[i], vector_clusters[j]))
            graph[j, i] = graph[i, j]

    node_weights = np.ones(num_nodes)
    PageRank(graph, node_weights)
    for i in range(len(k_mean_sentences)):
        k_mean_sentences[i].setWeightedPageRank(node_weights[i])
    k_mean_sentences = sorted(k_mean_sentences, key=lambda x: x.getWeightedPageRank(), reverse=True)
    index_max_between = 0
    max_between = 0
    for i in range(len(k_mean_sentences) - 1):
        current = k_mean_sentences[i].getWeightedPageRank() - k_mean_sentences[i + 1].getWeightedPageRank()
        if max_between < current:
            index_max_between = i
            max_between = current
    print(index_max_between, max_between)
    top_k_sentence_best = k_mean_sentences[: index_max_between+1]
    else_k_mean_sentences = k_mean_sentences[index_max_between + 1:]
    position_top_k_sentence_best = sorted(top_k_sentence_best, key=lambda x: x.getWeightedPosition(), reverse=True)
    summary = []
    sum_len = 0
    j = 0
    while sum_len < summary_length:
        if len(position_top_k_sentence_best) == 0:
            summary.append(else_k_mean_sentences[j])
            sum_len += len(else_k_mean_sentences[j].getPreProWords())
            j += 1
        elif len(position_top_k_sentence_best) > 1:
            maxxer_all = []
            while True:
                if len(position_top_k_sentence_best) == 1:
                    maxxer_all.append(position_top_k_sentence_best[0])
                    break
                between_aaaa = position_top_k_sentence_best[0].getWeightedPosition() - position_top_k_sentence_best[1].getWeightedPosition()
                if between_aaaa == 0:
                    maxxer_all.append(position_top_k_sentence_best[0])
                    position_top_k_sentence_best.pop(0)
                else:
                    maxxer_all.append(position_top_k_sentence_best[0])
                    break

            if len(maxxer_all) == 1:
                summary.append(position_top_k_sentence_best[0])
                sum_len += len(position_top_k_sentence_best[0].getPreProWords())
                position_top_k_sentence_best.pop(0)
            else:
                while maxxer_all and (sum_len < summary_length):
                    MMRval = {}
                    for sent in maxxer_all:
                        MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)
                    maxxer = max(MMRval, key=MMRval.get)
                    summary.append(maxxer)
                    maxxer_all.remove(maxxer)
                    sum_len += len(maxxer.getPreProWords())
                    summary.append(maxxer)
                position_top_k_sentence_best.pop(0)
        else:
            summary.append(position_top_k_sentence_best[0])
            sum_len += len(position_top_k_sentence_best[0].getPreProWords())
            position_top_k_sentence_best.pop(0)

    global system_nu, human_nu
    system_nu += sum_len
    human_nu += summary_length

    return summary

def makeSummaryPageRank(sentences, query, k_cluster, summary_length, lambta, IDF):
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
    vector_clusters = kmeans.cluster_centers_

    num_nodes = len(vector_clusters)
    graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # tinh toan độ trùng lặp giữa 2 sentences
            graph[i, j] = float(sim_cosin(vector_clusters[i], vector_clusters[j]))
            graph[j, i] = graph[i, j]

    node_weights = np.ones(num_nodes)
    PageRank(graph, node_weights)

    top_index = [i for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)]
    summary = []
    sum_len = 0

    # keeping adding sentences until number of words exceeds summary length
    print(top_index)
    for i in top_index:
        if (sum_len > n):
            break
        summary += [k_mean_sentences[i]]
        sum_len += len(k_mean_sentences[i].getPreProWords())

    global human_nu, system_nu
    human_nu += n
    system_nu += sum_len
    print(sum_len, n)

    return summary

def makeSummaryMMR(sentences, query, k_cluster, summary_length, lambta, IDF):
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

    best_sentence = bestSentence(k_mean_sentences, query, IDF)
    summary = [best_sentence]

    sum_len = len(best_sentence.getPreProWords())

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len <= summary_length) and k_mean_sentences:
        MMRval = {}

        for sent in k_mean_sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)
        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        k_mean_sentences.remove(maxxer)
        sum_len += len(maxxer.getPreProWords())

    global human_nu, system_nu
    human_nu += summary_length
    system_nu += sum_len
    print(sum_len, summary_length)
    return summary


def makeSummaryPositionMMR(sentences, query, k_cluster, n, lambta, IDF):
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
    position = [sen.getWeightedPosition() for sen in k_mean_sentences]
    summary = []
    sum_len = 0

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len < n) and k_mean_sentences:
        max_value = max(position)
        if position.count(max_value) == 1:
            maxxer = max(k_mean_sentences, key=lambda item: item.getWeightedPosition())
            summary.append(maxxer)
            k_mean_sentences.remove(maxxer)
            position = [sen.getWeightedPosition() for sen in k_mean_sentences]
            sum_len += len(maxxer.getPreProWords())
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
            sum_len += len(maxxer.getPreProWords())

    global human_nu, system_nu
    human_nu += n
    system_nu += sum_len
    return summary


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


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == '__main__':

    # set the main Document folder path where the subfolders are present
    main_folder_path = root_directory + "Data/DUC_2007/Documents"
    human_folder_path = root_directory + "Data/DUC_2007/Human_Summaries/"

    # read in all the subfolder names present in the main folder
    for folder in os.listdir(main_folder_path):

        print("Running Kmean Summarizer for files in folder: ", folder)
        # for each folder run the MMR summarizer and generate the final summary
        curr_folder = main_folder_path + "/" + folder

        file_human_1 = human_folder_path + "summary_" + folder[3:5] + ".A.1.txt"
        file_human_2 = human_folder_path + "summary_" + folder[3:5] + ".B.1.txt"
        file_human_3 = human_folder_path + "summary_" + folder[3:5] + ".C.1.txt"
        file_human_4 = human_folder_path + "summary_" + folder[3:5] + ".D.1.txt"
        text_1 = open(file_human_1, 'r').read()
        text_2 = open(file_human_2, 'r').read()
        text_3 = open(file_human_3, 'r').read()
        text_4 = open(file_human_4, 'r').read()
        n = 0
        for el in [text_1, text_2, text_3, text_4]:
            llll = nltk.word_tokenize(el)

            # stemming words // đưa về từ gốc
            stemmedSent = [porter.stem(word) for word in llll]
            stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
                                                     and x != '(' and x != ')' and x.find('&') == -1
                                                     and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                     and x != '``' and x != '--' and x != ':'
                                                     and x != "''" and x != "'s", stemmedSent))
            n += len(stemmedSent)
        n = n / 4

        sentences = []
        files = os.listdir(curr_folder)
        for file in files:
            sentences = sentences + processFile(curr_folder + "/" + file)
        k_cluster = getK_cluster(sentences, n)
        print(k_cluster)
        IDF_w = IDFs(sentences)
        TF_IDF_w = TF_IDF(sentences)
        query = buildQuery(sentences, TF_IDF_w, 10)

        # summary = makeSummaryMMR(sentences, query, k_cluster, n, 0.5, IDF_w)
        summary = makeSummaryPageRankPMMR(sentences, query, k_cluster, n, 0.5, IDF_w)
        # summary = makeSummaryPageRank(sentences, query, k_cluster, n, 0.5, IDF_w)
        # summary = makeSummaryPositionMMR(sentences, query, k_cluster, n, 0.5, IDF_w)

        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOriginalWords() + "\n"
        final_summary = final_summary[:-1]
        # results_folder = root_directory + "Data/DUC_2007/Kmean_results_position"
        # results_folder = root_directory + "Data/DUC_2007/Kmean_results_pagerank_position"
        results_folder = root_directory + "Data/DUC_2007/Kmean_results_pagerank_position_MMR"
        # results_folder = root_directory + "Data/DUC_2007/Kmean_results"
        with open(os.path.join(results_folder, (str(folder) + ".kmean")), "w") as fileOut:
            fileOut.write(final_summary)
    print(system_nu, human_nu)
