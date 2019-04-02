import nltk
import os
import math
import re
from method.English.MMR import sentence
from nltk.corpus import stopwords
nltk.download("stopwords")
chachedWords = stopwords.words('english')
human_nu = 0
system_nu = 0
porter = nltk.PorterStemmer()


# ---------------------------------------------------------------------------------
# Description	: Function to preprocess the files in the document cluster before
#				  passing them into the MMR summarizer system. Here the sentences
#				  of the document cluster are modelled as sentences after extracting
#				  from the files in the folder path. 
# Parameters	: file_name, name of the file in the document cluster
# Return 		: list of sentence object
# ---------------------------------------------------------------------------------
def processFile(file_name):
    # read file from provided folder path
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    f = open(file_name, 'r')
    text_0 = f.read()

    # extract content in TEXT tag and remove tags
    # code 2004
    # text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
    # text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
    # text_1 = re.sub("\n</TEXT>", "", text_1)
    #
    # # replace all types of quotations by normal quotes
    # text_1 = re.sub("\n", " ", text_1)
    #
    # text_1 = re.sub("\"", "\"", text_1)
    # text_1 = re.sub("''", "\"", text_1)
    # text_1 = re.sub("``", "\"", text_1)
    #
    # text_1 = re.sub(" +", " ", text_1)

    # code 2007
    text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
    text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
    text_1 = re.sub("\n</TEXT>", "", text_1)

    # replace all types of quotations by normal quotes
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

    # modelling each sentence in file as sentence object
    for line in lines:
        line = line.strip()
        # original words of the sentence before stemming // giữ lại câu gốc
        originalWords = line[:]

        # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
        line = line.lower()

        # word tokenization // tách từ
        sent = nltk.word_tokenize(line)

        # stemming words // đưa về từ gốc
        stemmedSent = [porter.stem(word) for word in sent]
        stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
                                            and x != '(' and x != ')' and x.find('&') == -1
                                            and x != '?' and x != "'" and x != '!' and x != '''"'''
                                            and x != '``' and x != '--' and x != ':'
                                            and x != "''" and x != "'s", stemmedSent))

        if (len(stemmedSent) <= 4):
            break
        # list of sentence objects
        if stemmedSent:
            sentences.append(sentence.sentence(file_name, stemmedSent, originalWords))

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
    idf = 0
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
    return sentence.sentence("query", queryWords, queryWords)


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
# Description	: Function to create the summary set of a desired number of words 
# Parameters	: sentences, sentences of the document cluster
#				  best_sentnece, best sentence in the document cluster
#				  query, reference query for the document cluster
#				  summary_length, desired number of words for the summary
#				  labmta, lambda value of the MMR score calculation formula
#				  IDF, IDF value of words in the document cluster 
# Return 		: name 
# ---------------------------------------------------------------------------------
def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):
    summary = [best_sentence]
    sum_len = len(best_sentence.getPreProWords())

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len < summary_length):
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        sentences.remove(maxxer)
        sum_len += len(maxxer.getPreProWords())
    global system_nu, human_nu
    system_nu += sum_len
    human_nu += summary_length

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


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == '__main__':
    root_directory = "/home/giangvu/Desktop/multi-summarization/"
    # set the main Document folder path where the subfolders are present
    # main_folder_path = os.getcwd() + "/DUC_2004/Documents"
    main_folder_path = root_directory + "Data/DUC_2007/Documents"
    human_folder_path = root_directory + "Data/DUC_2007/Human_Summaries/"
    # read in all the subfolder names present in the main folder
    for folder in os.listdir(main_folder_path):

        print("Running MMR Summarizer for files in folder: ", folder)
        # for each folder run the MMR summarizer and generate the final summary
        curr_folder = main_folder_path + "/" + folder

        # find all files in the sub folder selected
        files = os.listdir(curr_folder)

        file_human_1 = human_folder_path  + "summary_" + folder[3:5] + ".A.1.txt"
        file_human_2 = human_folder_path  + "summary_" + folder[3:5] + ".B.1.txt"
        file_human_3 = human_folder_path  + "summary_" + folder[3:5] + ".C.1.txt"
        file_human_4 = human_folder_path  + "summary_" + folder[3:5] + ".D.1.txt"
        text_1 = open(file_human_1, 'r').read()
        text_2 = open(file_human_2, 'r').read()
        text_3 = open(file_human_3, 'r').read()
        text_4 = open(file_human_4, 'r').read()
        length_summary = 0
        for el in [text_1, text_2, text_3, text_4]:
            llll = nltk.word_tokenize(el)

            # stemming words // đưa về từ gốc
            stemmedSent = [porter.stem(word) for word in llll]
            stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
                                                and x != '(' and x != ')' and x.find('&') == -1
                                                and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                and x != '``' and x != '--' and x != ':'
                                                and x != "''" and x != "'s", stemmedSent))
            length_summary += len(stemmedSent)
        length_summary = length_summary / 4

        sentences = []
        for file in files:
            sentences = sentences + processFile(curr_folder + "/" + file)

        # calculate TF, IDF and TF-IDF scores
        # TF_w 		= TFs(sentences)
        IDF_w = IDFs(sentences)

        TF_IDF_w = TF_IDF(sentences)

        # build query; set the number of words to include in our query
        query = buildQuery(sentences, TF_IDF_w, 10)

        # pick a sentence that best matches the query
        best1sentence = bestSentence(sentences, query, IDF_w)

        # build summary by adding more relevant sentences
        summary = makeSummary(sentences, best1sentence, query, length_summary, 0.5, IDF_w)

        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOriginalWords() + "\n"
        final_summary = final_summary[:-1]
        # results_folder = os.getcwd() + "/DUC_2004/MMR_results"
        results_folder = root_directory + "Data/DUC_2007/MMR_results"
        with open(os.path.join(results_folder, (str(folder) + ".MMR")), "w") as fileOut:
            fileOut.write(final_summary)
    print(system_nu, human_nu)