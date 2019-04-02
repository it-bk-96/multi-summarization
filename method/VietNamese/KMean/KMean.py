from __future__ import print_function
import os
import math
import nltk
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pyvi import ViTokenizer
import matplotlib.pyplot as plt

# nltk.download('punkt')
human_nu = 0
system_nu = 0


class sentence(object):

    def __init__(self, docName, preproWords, originalWords, weightedPosition):
        self.docName = docName
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.originalWords = originalWords
        self.weightedPosition = weightedPosition
        self.vectorSentence = self.caculateVector(preproWords)

    def caculateVector(self, preproWords):
        result = np.zeros(300)
        for word in preproWords:
            try:
                result += np.array(model_w2v[word])
            except:
                continue
        # result = result/len(preproWords)
        return result

    def getVectorSentence(self):
        return self.vectorSentence

    def getDocName(self):
        return self.docName

    def getPreProWords(self):
        return self.preproWords

    def getOriginalWords(self):
        return self.originalWords

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


def processFile(file_name):
    # read file from provided folder path
    f = open(file_name, 'r')
    text_1 = f.read()

    # tách câu
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = sentence_token.tokenize(text_1.strip())

    # setting the stemmer
    sentences = []

    # modelling each sentence in file as sentence object
    for i in range(len(lines)):
        line = lines[i]
        # giữ lại câu gốc
        originalWords = line[:]

        # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
        line = line.strip().lower()

        # tách từ
        stemmedSent = ViTokenizer.tokenize(line).split()

        stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                                            and x != '!' and x != '''"''' and x != "''" and x != '-',
                                  stemmedSent))

        if ((i + 1) == len(lines)) and (len(stemmedSent) <= 8):
            break
        # list of sentence objects
        if stemmedSent:
            sentences.append(sentence(file_name, stemmedSent, originalWords, float(1 / (i + 1))))

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


def makeSummaryPageRank(sentences, k_cluster, summary_length, IDF):
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


    # exit()
    # get k sentence nearest k cluster
    avg = []
    for j in range(k_cluster):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, A)
    ordering = sorted(range(k_cluster), key=lambda k: avg[k])

    k_mean_sentences = [sentences[closest[idx]] for idx in ordering]

    # page Rank
    # built matrix
    vector_clusters= kmeans.cluster_centers_
    num_nodes = len(vector_clusters)
    graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # tinh toan độ trùng lặp giữa 2 sentences
            graph[i, j] = float(sim_cosin(vector_clusters[i], vector_clusters[j]))
            graph[j, i] = graph[i, j]

    node_weights = np.ones(num_nodes)
    PageRank(graph, node_weights)

    summary = []
    top_index = [i for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)]
    current_length = 0

    for i in top_index:
        if (current_length > (summary_length - 20)):
            break
        summary += [k_mean_sentences[i]]
        current_length += len(ViTokenizer.tokenize(sentences[i].getOriginalWords()).split())
    print(current_length, summary_length)
    global human_nu, system_nu
    human_nu += summary_length
    system_nu += current_length

    return summary


def makeSummaryPosition(sentences, k_cluster, summary_length, IDF):
    # k mean
    vocabulary = []
    for sent in sentences:
        vocabulary = vocabulary + sent.getPreProWords()
    vocabulary = list(set(vocabulary))

    A = np.zeros(shape=(len(sentences), len(vocabulary)))
    for i in range(len(sentences)):
        for word in sentences[i].getWordFreq():
            index = vocabulary.index(word)
            A[i][index] = sentences[i].getWordFreq().get(word, 0) ** IDF[word]
    kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(A)

    avg = []
    for j in range(k_cluster):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, A)
    ordering = sorted(range(k_cluster), key=lambda k: avg[k])

    k_mean_sentences = [sentences[closest[idx]] for idx in ordering]

    # best_sentence = bestSentence(k_mean_sentences, query, IDF)
    summary = []

    # sum_len = len(best_sentence.getPreProWords())
    sum_len = 0

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len <= (summary_length - 25)):
        maxxer = max(k_mean_sentences, key=lambda item: item.getWeightedPosition())
        summary.append(maxxer)
        k_mean_sentences.remove(maxxer)
        sum_len += len(ViTokenizer.tokenize(maxxer.getOriginalWords()).split())
    global human_nu, system_nu
    human_nu += length_summary
    system_nu += sum_len
    return summary

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
    sum_len = 0
    position = [sen.getWeightedPosition() for sen in k_mean_sentences]

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len <= (summary_length - 25)) and k_mean_sentences:
        max_value = max(position)
        print(max_value, position.count(max_value))
        if position.count(max_value) == 1:
            maxxer = max(k_mean_sentences, key=lambda item: item.getWeightedPosition())
            summary.append(maxxer)
            k_mean_sentences.remove(maxxer)
            position = [sen.getWeightedPosition() for sen in k_mean_sentences]
            sum_len += len(maxxer.getPreProWords())
        else:
            MMRval = {}
            list_p = []
            print('vao dc nay')
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
            sum_len += len(ViTokenizer.tokenize(maxxer.getOriginalWords()).split())

    global human_nu, system_nu
    human_nu += length_summary
    system_nu += sum_len
    return summary


def makeSummaryMMR(sentences, query, k_cluster, summary_length, lambta, IDF):
    # k mean
    vocabulary = []
    for sent in sentences:
        vocabulary = vocabulary + sent.getPreProWords()
    vocabulary = list(set(vocabulary))

    A = np.zeros(shape=(len(sentences), len(vocabulary)))
    for i in range(len(sentences)):
        for word in sentences[i].getWordFreq():
            index = vocabulary.index(word)
            A[i][index] = sentences[i].getWordFreq().get(word, 0) ** IDF[word]
    kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(A)

    avg = []
    for j in range(k_cluster):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, A)
    ordering = sorted(range(k_cluster), key=lambda k: avg[k])

    k_mean_sentences = [sentences[closest[idx]] for idx in ordering]

    best_sentence = bestSentence(k_mean_sentences, query, IDF)
    summary = [best_sentence]

    sum_len = len(ViTokenizer.tokenize(best_sentence.getOriginalWords()).split())

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len <= (summary_length - 20)):
        MMRval = {}

        for sent in k_mean_sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)
        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        k_mean_sentences.remove(maxxer)
        sum_len += len(ViTokenizer.tokenize(maxxer.getOriginalWords()).split())

    global human_nu, system_nu
    human_nu += length_summary
    system_nu += sum_len
    return summary


def makeSummaryMMRW2V(sentences, query, k_cluster, summary_length, lambta, IDF):
    # k mean
    A = []
    for i in range(len(sentences)):
        A.append(sentences[i].getVectorSentence())

    kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(np.array(A))

    avg = []
    for j in range(k_cluster):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, A)
    ordering = sorted(range(k_cluster), key=lambda k: avg[k])

    k_mean_sentences = [sentences[closest[idx]] for idx in ordering]

    best_sentence = bestSentence(k_mean_sentences, query, IDF)
    summary = [best_sentence]

    sum_len = len(ViTokenizer.tokenize(best_sentence.getOriginalWords()).split())

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len <= (summary_length - 25)) and k_mean_sentences:
        MMRval = {}

        for sent in k_mean_sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)
        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        k_mean_sentences.remove(maxxer)
        sum_len += len(ViTokenizer.tokenize(maxxer.getOriginalWords()).split())

    global human_nu, system_nu
    human_nu += summary_length
    system_nu += sum_len
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


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == '__main__':
    root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/"

    with open(root_directory + "models/w2v.model", 'rb') as f:
        model_w2v = pickle.load(f)
    # set the main Document folder path where the subfolders are present
    main_folder_path = root_directory + "Data/VietNamese/Documents"
    human_folder_path = root_directory + "Data/VietNamese/Human_Summaries/"

    stop_word = list(map(lambda x: "_".join(x.split()),
                         open(root_directory + "vietnamese-stopwords.txt", 'r').read().split(
                             "\n")))
    # read in all the subfolder names present in the main folder
    for folder in os.listdir(main_folder_path):

        print("Running Kmean Summarizer for files in folder: ", folder)
        # for each folder run the MMR summarizer and generate the final summary
        curr_folder = main_folder_path + "/" + folder

        # find all files in the sub folder selected
        files = os.listdir(curr_folder)

        file_human_1 = human_folder_path + folder + ".ref1.txt"
        text_1_token = open(file_human_1, 'r').read()
        text_1_token = text_1_token.strip().lower()
        text_1_token = ViTokenizer.tokenize(text_1_token).split()

        file_human_2 = human_folder_path + folder + ".ref2.txt"
        text_2_token = open(file_human_2, 'r').read()
        text_2_token = text_2_token.strip().lower()
        text_2_token = ViTokenizer.tokenize(text_2_token).split()
        length_summary = (len(text_1_token) + len(text_2_token)) // 2

        sentences = []

        for file in files:
            sentences = sentences + processFile(curr_folder + "/" + file)

        # calculate TF, IDF and TF-IDF scores
        IDF_w = IDFs(sentences)
        TF_IDF_w = TF_IDF(sentences)
        k_cluster = getK_cluster(sentences, length_summary)
        print(k_cluster, length_summary, len(sentences))
        # continue
        # build query; set the number of words to include in our query
        query = buildQuery(sentences, TF_IDF_w, 10)

        # build summary by adding more relevant sentences
        summary = makeSummaryPageRank(sentences, k_cluster, length_summary, IDF_w)
        # summary = makeSummaryPosition(sentences, k_cluster, length_summary, IDF_w)
        # summary = makeSummaryPositionMMR(sentences, query, k_cluster, length_summary, 0.5, IDF_w)
        # summary = makeSummaryMMRW2V(sentences, query, k_cluster, length_summary, 0.5, IDF_w)
        # summary = makeSummaryMMR(sentences, query, k_cluster, length_summary, 0.5, IDF_w)

        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOriginalWords() + "\n"
        final_summary = final_summary[:-1]
        # results_folder = root_directory + "Data/VietNamese/K_mean_results"
        # results_folder = root_directory + "Data/VietNamese/K_mean_results_W2V"
        # results_folder = root_directory + "Data/VietNamese/K_mean_results_Position"
        results_folder = root_directory + "Data/VietNamese/K_mean_results_PageRank"
        # results_folder = root_directory + "Data/VietNamese/K_mean_results_Position_MMR"
        with open(os.path.join(results_folder, (str(folder) + ".kmean")), "w") as fileOut:
            fileOut.write(final_summary)
    print(system_nu, human_nu)
