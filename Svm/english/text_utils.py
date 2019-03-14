import nltk
import operator
import os
import re
import json
import math
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

SPECICAL_CHARACTER = {'"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
QUOTE = {'"'}


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


def split_sent_notoken(file_name):
    with open(file_name, 'r') as file:
        sentences = file.read().split('\n')
    return sentences


def split_sentences(file_name):
    sentences = []
    with open(file_name, 'r') as file:
        data = file.read().split('\n')
        for sent in data:
            sentences.append(sent)  # remove stopwords

    # sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    # tmp = sentence_token.tokenize(text_system)

    # sentences = []
    # for item in tmp:
    #     if "…" in item:
    #         b = item.split("…")
    #         for i in b:
    #             sentences.append(i)
    #     else:
    #         sentences.append(item)

    return sentences


def split_sentences_from_text(text):
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    tmp = sentence_token.tokenize(text)

    sentences = []
    for item in tmp:
        if "…" in item:
            b = item.split("…")
            for i in b:
                sentences.append(i)
        else:
            sentences.append(item)

    return sentences


def separate_label_sent(sentences):
    arr_labels = []
    arr_sents = []
    for i in sentences:
        arr_labels.append(i[0])
        arr_sents.append(i[2:])
    return arr_labels, arr_sents


def remove_short_sents(old_sentences):
    new_sentences_stem = []
    new_sentences_origin = []
    for i in range(len(old_sentences)):
        line = old_sentences[i]

        # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
        stemmedSent = line.strip().split()  # remove label => line[2:]

        stemmedSent = list(
            filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                             and x != '!' and x != '''"''' and x != "''" and x != '-',
                   stemmedSent))

        if ((i + 1) == len(old_sentences)) and (len(stemmedSent) <= 5):
            break
        if stemmedSent != []:
            new_sentences_stem.append(stemmedSent)
            new_sentences_origin.append(old_sentences[i])

    return new_sentences_stem, new_sentences_origin


# def text_process(sentences, stop_words):
#     new_sentences = []
#
#     for item in sentences:
#         tmp = item.lower()
#         text_tmp = ""
#         for word in tmp.split(' '):
#             if (word not in stop_words) and (len(word) != 1 or word in SPECICAL_CHARACTER):
#                 text_tmp += word + " "
#
#         new_sentences.append(text_tmp[:-1])
#
#     return new_sentences

def text_process(sentences, stop_words):
    match = {
        ' not': 'n\'t',
        '': '(\'s|\'ll|\'re|\'d|\'ve)',
        ' ': '[^a-zA-Z0-9"]',
        ' " ': '"'
    }

    for key in match:
        sentences = [sent[0:2] + re.sub(match[key], key, sent[2:]) for sent in sentences]

    new_sentences = []

    for item in sentences:
        text_tmp = ""
        for word in item.lower().split(' '):
            l = len(word)
            if (word not in stop_words) and l != 0:
                text_tmp += word + " "
        new_sentences.append(text_tmp)

    return new_sentences


# def text_process_all(sentences, stop_words):
#     new_sentences = []
#
#     match = {
#         ' not': 'n\'t',
#         '': '(\'s|\'ll|\'re|\'d|\'ve)',
#         ' ': '[^a-zA-Z0-9]'
#     }
#
#     for key in match:
#         # remove label from data => [2:]
#         sentences = [re.sub(match[key], key, sent[2:]) for sent in sentences]
#
#     for item in sentences:
#         tmp = item.strip().lower()
#         text_tmp = []
#         for word in tmp.split(' '):
#             if (word not in stop_words) and (len(word) != 1 or word in SPECICAL_CHARACTER):
#                 text_tmp.append(word)
#
#         new_sentences.append(' '.join(text_tmp))
#
#     return new_sentences


def text_process_all(sentences, stop_words):
    new_sentences = []
    lematizer = WordNetLemmatizer()
    stemmer = LancasterStemmer()
    match = {
        ' not': 'n\'t',
        '': '(\'s|\'ll|\'re|\'d|\'ve)',
        ' ': '[^a-zA-Z0-9]'
    }

    sentences_nolabel = [sent[2:] for sent in sentences]
    for key in match:
        sentences_nolabel = [re.sub(match[key], key, sent) for sent in sentences_nolabel]

    for item in sentences_nolabel:
        tmp = item.strip().lower()  # remove label from data => [2:]
        text_tmp = []
        for word in tmp.split(' '):
            l = len(word)
            if (word not in stop_words) and l != 0:
                word = stemmer.stem(lematizer.lemmatize(word))
                text_tmp.append(word)


        new_sentences.append(' '.join(text_tmp))

    return new_sentences


def read_stopwords(file_name):
    with open(file_name, 'r') as file:
        tmp_stop_words = []
        for w in file.readlines():
            tmp_stop_words.append(w.strip().replace(' ', '_'))

        stop_words = []
        for item in tmp_stop_words:
            if item != '':
                stop_words.append(item)
        stopwords = set(stop_words)
    return stopwords


def get_sentence_first_paragraph(file_name, stop_words):
    with open(file_name, 'r') as file:
        text = file.readlines()

    contain = []
    for item in text:
        contain.append(split_sentences_from_text(item)[0])

    contain = text_process(contain, stop_words)

    contain = set(contain)

    return contain


def get_doc_from_sentences(sentences):
    doc = ""
    for item in sentences:
        doc += item + " "

    return doc[:-1]


def get_dict_words_from_doc(document):
    words = {}

    for word in document.split(' '):
        if word not in QUOTE:
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1

    return words


def get_word_freq(word, document):
    dict_words = get_dict_words_from_doc(document)

    if word not in dict_words:
        return 0

    return dict_words[word]


def tf(word, document):
    dict_words = get_dict_words_from_doc(document)
    count_words = 0

    for item in dict_words:
        count_words += dict_words[item]

    return dict_words[word] * 1.0 / count_words


def idf(word, documents):
    N = len(documents)

    contain_word = 0
    for item in documents:

        if word in item.split(' '):
            contain_word += 1

    return math.log(1.0 * N / contain_word)


def save_idf(idf_dict, output):
    with open(output, 'w') as fp:
        json.dump(idf_dict, fp)
    fp.close()

def read_json_file(path):
    with open(path, 'r') as fp:
        dicti = json.load(fp)
    fp.close()
    return dicti


def get_all_idf(documents):
    words = {}

    for item in documents:
        for word in item.split(' '):
            if word not in QUOTE and word not in words:
                words[word] = 0.0

    for item in words:
        words[item] = idf(item, documents)

    return words


def get_freq_word_uni(document):
    tf_words = {}
    for item in document.split(" "):
        if item not in QUOTE and item not in tf_words:
            tf_words[item] = tf(item, document)

    number_freq_word = int(0.3 * len(tf_words))
    freq_words = {}

    i = 0
    for key, value in sorted(tf_words.items(), key=operator.itemgetter(1), reverse=True):
        freq_words[key] = value

        i += 1
        if i == number_freq_word:
            break

    return freq_words


def get_centroid_uni(document, all_idf):
    words = {}

    for word in document.split(' '):
        if word not in QUOTE and word not in words:
            words[word] = 0.0

    for item in words:
        if item in all_idf:
            words[item] = tf(item, document) * all_idf[item]
        else:
            words[item] = tf(item, document) * 0.1

    number_centroid_uni = int(0.3 * len(words))

    centroid_uni = {}
    i = 0
    for key, value in sorted(words.items(), key=operator.itemgetter(1), reverse=True):
        centroid_uni[key] = value

        i += 1
        if i == number_centroid_uni:
            break

    return centroid_uni

def read_all_documents(file_names, stop_words):
    documents = []
    for item in file_names:
        sentences = split_sent_notoken(item)
        sentences = text_process_all(sentences, stop_words)  # remove stopwords

        sentences_not_short = remove_short_sents(sentences)[1]  # remove short sents
        documents.append(get_doc_from_sentences(sentences_not_short))

    return documents


def convert_uni_to_bi(documents):
    bi_documents = []

    for item in documents:
        words = []
        for word in item.split(' '):
            if word not in QUOTE:
                words.append(word)

        bi_document = []
        for item in range(1, len(words)):
            bi_document.append(words[item - 1] + "__" + words[item])

        tmp = ""
        for item in bi_document:
            tmp += item + " "

        bi_documents.append(tmp[:-1])

    return bi_documents


def cos_similarity(s1, idf, sentences):
    '''
    compute cosine similarity of any sentence with first sentence of once document
    :param s1:
    :param s2:
    :param list_sent:
    :return:
    '''

    doc = get_doc_from_sentences(sentences)
    numerator = 0
    denom1 = 0
    denom2 = 0
    list_word_s1_tmp = s1.split(' ')
    list_word_s2_tmp = sentences[0].split(' ')

    list_word_s1 = []
    for item in list_word_s1_tmp:
        if item not in QUOTE:
            list_word_s1.append(item)

    list_word_s2 = []
    for item in list_word_s2_tmp:
        if item not in QUOTE:
            list_word_s2.append(item)

    all_words = set(list_word_s1 + list_word_s2)

    tf_arr = {}
    for word in all_words:
        tf_arr[word] = tf(word, doc)
    for word in list_word_s1:
        if word in idf:
            denom1 += (tf_arr[word] * idf[word]) ** 2
        else:
            denom1 += (tf_arr[word] * 0.1) ** 2

    for word in list_word_s2:
        tf_w = tf_arr[word]
        idf_w = 0
        if word in idf:
            idf_w = idf[word]
            denom2 += (tf_w * idf_w) ** 2
        if word in list_word_s1 and word in idf:
            numerator += (tf_w * idf_w) ** 2
        else:
            numerator += (tf_w * 0.1) ** 2
            denom2 += (tf_w * 0.1) ** 2
    sim = 0.0
    try:
        sim = numerator / (math.sqrt(denom1) * math.sqrt(denom2))
    except Exception:
        pass

    return sim


# def prepare_data_svm(ar_labels, ar_svm, ar_nmf, ar_lexrank, output):
#     all_features = []
#     for i in range(len(ar_labels)):
#         feature = ar_labels[i]
#         vector = ar_svm[i]
#         for i in range(len(vector)):
#             feature += ' ' + str(i + 1) + ':' + str(vector[i])
#         feature += ' 10:' + str(ar_nmf[i]) + ' 11:' + str(ar_lexrank[i])
#         all_features.append(feature)
#
#     with open(output, 'w') as f:
#         f.write('\n'.join(all_features))
#         f.close()

def prepare_data_svm(ar_labels, ar_svm, output):
    all_features = []
    for i in range(len(ar_labels)):
        feature = ar_labels[i] + ' ' + ' '.join(list(map(str, ar_svm[i])))
        # feature += ' ' + str(ar_nmf[i]) + ' ' + str(ar_lexrank[i])
        all_features.append(feature)

    with open(output, 'w') as f:
        f.write('\n'.join(all_features))
        f.close()


# def prepare_data_svm(ar_labels, ar_svm, output):
#     all_features = []
#     for i in range(len(ar_labels)):
#         feature = []
#         feature += ar_labels[i]
#         feature += list(map(str, ar_svm[i]))
#         #feature += ' ' + str(ar_nmf[i]) + ' ' + str(ar_lexrank[i])
#         all_features.append(feature)
#
#     # with open(output, 'w') as f:
#     #     f.write('\n'.join(all_features))
#     #     f.close()
#     np.save(output, np.array(all_features))


def normalize(ar_ar_numbers):
    all_figures = []
    for numbers in ar_ar_numbers:
        max_number = max(numbers)
        normalized_numbers = []

        for number in numbers:
            normalized_numbers.append(number / max_number)
        all_figures.append(normalized_numbers)

    return all_figures


def write_file_text(data, path_file):
    out = open(path_file, 'w')
    out.write(data)
    out.close()


def read_file_text(path_file):
    with open(path_file, 'r') as content:
        data = content.read()
    content.close()
    return data


def convert_features_svm(path):
    train = []
    test = []
    for clus in os.listdir(path + '/' + 'train'):
        f = open(path + '/train/' + clus, 'r')
        train += f.read().split('\n')
        f.close()

    for clus in os.listdir(path + '/' + 'test'):
        t = open(path + '/test/' + clus, 'r')
        test += t.read().split('\n')
        t.close()
    # train, test = train_test_split(all_features, test_size=0.2,random_state= 42)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for vec in train:
        X_train.append(list(map(float, vec[2:].split(' '))))  # [:9]
        Y_train.append(int(vec[0]))

    for v in test:
        X_test.append(list(map(float, v[2:].split(' '))))  # [:9]
        Y_test.append(int(v[0]))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test
# if __name__ == "__main__":
#     sentences = split_sentences("/home/hieupd/PycharmProjects/multi_summari_svm/Data_Non_Token/Documents/cluster_1/12240106.body.txt")
#     sentences2 = split_sentences("/home/hieupd/PycharmProjects/multi_summari_svm/Data_Non_Token/Documents/cluster_1/12240586.body.txt")
#     # # print (sentences)
#     #
#     # # print (sentences)
#     #
#     stop_words = read_stopwords("/home/hieupd/PycharmProjects/multi_summari_svm/stopwords.txt")
#     # # print (stop_words)
#     #
#     sentences = text_process(sentences, stop_words)
#     print(sentences)
#     doc = get_doc_from_sentences(sentences)
#
#     sentences2 = text_process(sentences2, stop_words)
#     doc2 = get_doc_from_sentences(sentences2)
#
#     # print (tf("Người", doc))
#     a = get_freq_word_uni(doc)
#
#     # print (a)
#     # a = convert_uni_to_bi([doc])
#     #
#     # print (a)
#
#     all_idf = get_all_idf([doc, doc2])
#     print(all_idf)
#
#     print(cos_similarity(sentences[1], all_idf, sentences))
#
#     # print (all_idf)
#     #
#     # a = get_centroid_uni(doc, all_idf)
#     #
#     # print (a)
#     # print (get_all_word([sentences, sentences]))
#     # for item in sentences:
#     #     print (item)
#     # print ("----------------------------------")
#     # contain = get_sentence_first_paragraph("document1.txt", stop_words)
#     #
#     # for item in sentences:
#     #     if item in contain:
#     #         print (1)
#     #     else:
#     #
#     #         print (0)
