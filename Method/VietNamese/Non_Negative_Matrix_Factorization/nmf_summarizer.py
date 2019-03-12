import os
import math
import nltk
import pickle
import numpy as np
from pyvi import ViTokenizer
from sklearn.decomposition import NMF

human_nu = 0
system_nu = 0

root_directory = "/home/giangvu/Desktop/multi-summarization/"

with open(root_directory + "models/w2v.model", 'rb') as f:
    model_w2v = pickle.load(f)


class sentence(object):

    # ------------------------------------------------------------------------------
    # Description	: Constructor to initialize the setence object
    # Parameters  	: docName, name of the document/file
    #				  preproWords, words of the file after the stemming process
    #				  originalWords, actual words before stemming
    # Return 		: None
    # ------------------------------------------------------------------------------
    def __init__(self, docName, preproWords, originalWords):
        self.docName = docName
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.originalWords = originalWords
        self.score = 0
        self.scoreMMR = 0
        self.scoreFinal = 0
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

    def setScoreMMR(self, score):
        self.scoreMMR = score

    def getScoreMMR(self):
        return self.scoreMMR

    def setScoreFinal(self, score):
        self.scoreFinal = score

    def getScoreFinal(self):
        return self.scoreFinal

    # ------------------------------------------------------------------------------
    # Description	: Function to return the name of the document
    # Parameters	: None
    # Return 		: name of the document
    # ------------------------------------------------------------------------------
    def getDocName(self):
        return self.docName

    # ------------------------------------------------------------------------------
    # Description	: Function to return the stemmed words
    # Parameters	: None
    # Return 		: stemmed words of the sentence
    # ------------------------------------------------------------------------------
    def getPreProWords(self):
        return self.preproWords

    # ------------------------------------------------------------------------------
    # Description	: Function to return the original words of the sentence before
    #				  stemming
    # Parameters	: None
    # Return 		: pre-stemmed words
    # ------------------------------------------------------------------------------
    def getOriginalWords(self):
        return self.originalWords

    # ------------------------------------------------------------------------------
    # Description	: Function to return a dictonary of the word frequencies for
    #				  the particular sentence object
    # Parameters	: None
    # Return 		: dictionar of word frequencies
    # ------------------------------------------------------------------------------
    def getWordFreq(self):
        return self.wordFrequencies

    # ------------------------------------------------------------------------------
    # Description	: Function to create a dictonary of word frequencies for the
    #				  sentence object
    # Parameters	: None
    # Return 		: dictionar of word frequencies
    # ------------------------------------------------------------------------------
    def sentenceWordFreq(self):
        wordFreq = {}
        for word in self.preproWords:
            if word not in wordFreq.keys():
                wordFreq[word] = 1
            else:
                wordFreq[word] += + 1

        return wordFreq


# nltk.download('punkt')

# ---------------------------------------------------------------------------------
# Description	: Function to preprocess the files in the document cluster before
#				  passing them into the NFM summarizer system. Here the sentences
#				  of the document cluster are modelled as sentences after extracting
#				  from the files in the folder path. 
# Parameters	: file_name, name of the file in the document cluster
# Return 		: list of sentence object
# ---------------------------------------------------------------------------------
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
                                            and x != '!' and x != '''"''' and x != "''" and x != '-' and x not in stop_word,
                                  stemmedSent))

        if ((i + 1) == len(lines)) and (len(stemmedSent) <= 5):
            break
        # list of sentence objects
        if stemmedSent != []:
            sent = sentence(file_name, stemmedSent, originalWords)
            sentences.append(sent)

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
    # Method variables
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
    return sentence("query", queryWords, queryWords)


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


# ---------------------------------------------------------------------------------
# Description	: Function to create the summary set of a desired number of words
# Parameters	: sentences, sentences of the document cluster
#				  best_sentnece, best sentence in the document cluster
#				  query, reference query for the document cluster
#				  summary_length, desired number of words for the summary
#				  labmta, lambda value of the MMR score calculation formula
#				  IDF, IDF value of words in the document cluster
# Return 		: name
# ---------------------------------------------------------------------------------


def makeSummary(sentences, n):
    scores = np.sum(H, axis=0)
    max_score = max(scores)
    for i in range(len(sentences)):
        sentences[i].setScore(scores[i] / max_score)
    sentences = sorted(sentences, key=lambda x: x.getScore(), reverse=True)
    summary = []
    i = 0
    length_summary = len(ViTokenizer.tokenize(sentences[i].getOriginalWords().strip()).split())
    while (length_summary < n):
        i += 1
        summary += [sentences[i]]
        length_summary += len(ViTokenizer.tokenize(sentences[i].getOriginalWords().strip()).split())

    return summary


def makeSummary_Gong_Liu(sentences, n, H):
    i = 0
    index_H = np.argmax(H[i])
    length_summary = len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())
    array_index = []
    array_index.append(index_H)
    while (length_summary < (n - 20)):
        i += 1
        index_H = np.argmax(H[i])
        if index_H not in array_index:
            array_index.append(index_H)
            length_summary += len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())

    global system_nu, human_nu
    human_nu += n
    system_nu += length_summary
    return [sentences[index] for index in array_index]


def makeSummary_Topic(sentences, n, H):
    length_sentences = len(sentences)
    length_concept = len(H)
    average_concept = [sum(H[i][j] for j in range(length_sentences)) / length_sentences for i in range(length_concept)]
    for i in range(length_concept):
        for j in range(length_sentences):
            if H[i][j] < average_concept[i]:
                H[i][j] = 0
    matrix_conceptxconcept = [[0 for _ in range(length_concept)] for _ in range(length_concept)]

    for i in range(length_concept):
        for j in range(length_concept):
            if matrix_conceptxconcept[j][i] != 0:
                matrix_conceptxconcept[i][j] = matrix_conceptxconcept[j][i]
                continue
            total = 0
            for k in range(length_sentences):
                if (H[i][k] != 0) and (H[j][k] != 0):
                    total += (H[i][k] + H[j][k])
            matrix_conceptxconcept[i][j] = total

    strength_concept = [sum(matrix_conceptxconcept[i][j] for j in range(length_concept)) for i in range(length_concept)]
    top_index = np.array(strength_concept).argsort()[-length_concept:][::-1]
    i = 0
    index_H = np.argmax(H[top_index[i]])
    length_summary = len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())
    array_index = []
    array_index.append(index_H)
    while (length_summary < n):
        i += 1
        index_H = np.argmax(H[top_index[i]])
        if index_H not in array_index:
            array_index.append(index_H)
            length_summary += len(ViTokenizer.tokenize(sentences[index_H].getOriginalWords().strip()).split())

    return [sentences[index] for index in array_index]


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == '__main__':

    # set the main Document folder path where the subfolders are present
    main_folder_path = root_directory + "Data/Data_VietNamese/Documents"
    human_folder_path = root_directory + "Data/Data_VietNamese/Human_Summaries/"

    stop_word = list(map(lambda x: "_".join(x.split()),
                         open(root_directory + "vietnamese-stopwords.txt", 'r').read().split(
                             "\n")))

    # read in all the subfolder names present in the main folder
    for folder in os.listdir(main_folder_path):
        # start_time = time.time()

        print("Running NMF Summarizer for files in folder: ", folder)
        # for each folder run the MMR summarizer and generate the final summary
        curr_folder = main_folder_path + "/" + folder

        # find all files in the sub folder selected
        files = os.listdir(curr_folder)

        file_human_1 = human_folder_path + folder + ".ref1.txt"
        file_human_2 = human_folder_path + folder + ".ref2.txt"
        text_1 = open(file_human_1, 'r').read()
        text_2 = open(file_human_2, 'r').read()
        text_1_token = ViTokenizer.tokenize(text_1)
        text_2_token = ViTokenizer.tokenize(text_2)
        length_summary = int((len(text_1_token.split()) + len(text_1_token.split())) / 2)

        sentences = []

        for file in files:
            sentences = sentences + processFile(curr_folder + "/" + file)

        # calculate TF, IDF and TF-IDF scores
        # TF_w 		= TFs(sentences)
        # IDF_w = IDFs(sentences)
        # TF_IDF_w = TF_IDF(sentences)
        vocabulary = []
        for sent in sentences:
            vocabulary = vocabulary + sent.getPreProWords()
        vocabulary = list(set(vocabulary))
        A = np.zeros(shape=(len(vocabulary), len(sentences)))

        # tf
        for i in range(len(sentences)):
            tf_sentence = sentences[i].getWordFreq()
            for word in tf_sentence.keys():
                index = vocabulary.index(word)
                A[index][i] += tf_sentence[word]

        # binary
        # for i in range(len(sentences)):
        # 	preproWord = sentences[i].getPreProWords()
        # 	for word in preproWord:
        # 		index = vocabulary.index(word)
        # 		A[index][i] = 1

        # A = np.zeros(shape=(300, len(sentences)))
        # for i in range(len(sentences)):
        # 	vector = sentences[i].getVectorSentence()
        # 	for j in range(300):
        # 		if vector[j] < 0:
        # 			A[j][i] = 0
        # 		else:
        # 			A[j][i] = vector[j]

        rank_A = np.linalg.matrix_rank(A)

        print(rank_A)
        model = NMF(n_components=rank_A, init='random', random_state=0)
        W = model.fit_transform(A)
        H = model.components_

        # build summary
        summary = makeSummary_Gong_Liu(sentences, length_summary, H)
        # summary = makeSummary(sentences, length_summary)
        # summary = makeSummary_Topic(sentences, length_summary, H)

        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOriginalWords() + "\n"
        final_summary = final_summary[:-1]
        results_folder = root_directory + "Data/Data_VietNamese/NMF_results_w2v"
        with open(os.path.join(results_folder, (str(folder) + ".NMF")), "w") as fileOut:
            fileOut.write(final_summary)
    print(human_nu, system_nu)
