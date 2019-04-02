import math
from apiSummarization.Method.MMR import sentence
import nltk
import os

nltk.download('punkt')
from pyvi import ViTokenizer

root_abtract = os.path.dirname(os.getcwd())


def processFileVietNamese(documents):
    # Đọc file
    stop_word = list(map(lambda x: "_".join(x.split()),
                         open(root_abtract + "/vietnamese-stopwords.txt",
                              'r').read().split("\n")))

    sentences = []
    for document in documents:

        text_0 = document

        # tách câu
        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
        lines = sentence_token.tokenize(text_0.strip())

        # modelling each sentence in file as sentence object
        for i in range(len(lines)):
            line = lines[i]
            # giữ lại câu gốc
            OG_sent = line[:]

            # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
            line = line.strip().lower()

            # tách từ
            stemmed_sentence = ViTokenizer.tokenize(line).split()
            stemmed_sentence = list(
                filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                                 and x != '!' and x != '''"''' and x != "''" and x != '-' and x != '>>' and x not in stop_word,
                       stemmed_sentence))
            if ((i + 1) == len(lines)) and (len(stemmed_sentence) <= 5):
                break
            if stemmed_sentence != []:
                sentences.append(sentence.sentence(stemmed_sentence, OG_sent))
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
    return sentence.sentence(queryWords, queryWords)


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


def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):
    summary = [best_sentence]

    sum_len = len(ViTokenizer.tokenize(best_sentence.getOriginalWords()).split())

    while (sum_len <= summary_length):
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        sentences.remove(maxxer)
        sum_len += len(ViTokenizer.tokenize(maxxer.getOriginalWords()).split())

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


def getMMR(documents, length_summary):
    sentences = processFileVietNamese(documents)
    IDF_w = IDFs(sentences)
    TF_IDF_w = TF_IDF(sentences)
    query = buildQuery(sentences, TF_IDF_w, 10)

    best1sentence = bestSentence(sentences, query, IDF_w)

    summary = makeSummary(sentences, best1sentence, query, length_summary, 0.5, IDF_w)

    final_summary = ""
    for sent in summary:
        final_summary = final_summary + sent.getOriginalWords() + "\n"
    final_summary = final_summary[:-1]

    return final_summary


if __name__ == '__main__':
    path1 = root_abtract + '/Data/Documents/cluster_1/12240106.body.txt'
    path2 = root_abtract + '/Data/Documents/cluster_1/12240586.body.txt'
    path3 = root_abtract + '/Data/Documents/cluster_1/12241528.body.txt'
    text_1 = open(path1, 'r').read()
    text_2 = open(path2, 'r').read()
    text_3 = open(path3, 'r').read()
    print(getMMR([text_1, text_2, text_3], 100))
