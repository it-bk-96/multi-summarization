from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer
import os
import math
from data_interact import write_arr_string
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

TOP_TF_UNI = 'Top_TF_Uni'
TOP_TF_BI = 'Top_TF_Bi'
DATA_200_CLUSTERS = 'Data_Non_Token/Documents'
STOPWORDS = 'stopwords.txt'
CLUSTERS_TOKEN = 'Data_Token'
DATA_TOKEN_WORDS = 'Data_Token_Words'
DATA_REMOVED_STOPWORDS = 'Data_Remove_Stopwords'
DATA_BIGRAM = 'Data_Bigram'
SENT_SIMILARITIES = 'Sent_Similarities'
TFIDF_UNI = 'Tfidf_Uni'
FIRST_SENT_DOC = 'First_Sentence'


def tf(word, doc):
    count_w = doc.count(word)
    count_word_doc = len(doc.split(' '))

    return count_w / count_word_doc


def count_num_doc_contain(w, listdoc):
    '''
    count num doc contain a word
    :param w:
    :param listdoc:
    :return:
    '''
    num = 0
    for doc in listdoc:
        if w in doc.split(' '):
            num += 1
    return num


def idf(word, list_doc):
    num_doc = len(list_doc)
    return math.log(num_doc / count_num_doc_contain(word, list_doc))


def tfidf(word, doc, list_doc):
    return tf(word, doc) * idf(word, list_doc)


# get all document to calculate tfidf for each word in sentence in doc
def get_all_doc(path):
    all_docs = []
    for filename in os.listdir(DATA_REMOVED_STOPWORDS):
        with open(DATA_REMOVED_STOPWORDS + '/' + filename) as f:
            all_docs.append(f.read())

    return all_docs


# def cal_tfidf_alldoc():
#
#     '''
#     compute tfidf of all document to
#     :return:
#     '''
#
#     for filename in os.listdir(DATA_REMOVED_STOPWORDS):
#         print("cal", filename)
#         arr_all = []
#         with open(DATA_REMOVED_STOPWORDS + '/' + filename) as f:
#             list_sent = f.read().split('\n')
#
#             for sent in list_sent:
#                 arr_tfidf = []
#                 list_words = set(sent.split(' '))
#
#                 for w in list_words:
#                     arr_tfidf.append((w, tfidf(w, sent, list_sent)))
#                 arr_tfidf_sorted = sorted(arr_tfidf, key=lambda x: x[1], reverse=True)
#                 arr_tfidf_top = []
#
#                 for word, score in arr_tfidf_sorted[:len(arr_tfidf)]:
#                     score = round(score, 6)
#                     arr_tfidf_top.append(word + ':' + str(score))
#
#                 arr_all.append(','.join(arr_tfidf_top))
#
#         write_arr_string(arr_all, TFIDF_UNI + '/' + filename)


def cos_similarity(s1, s2, list_sent):
    '''
    compute cosine similarity of any sentence with first sentence of once document
    :param s1:
    :param s2:
    :param list_sent:
    :return:
    '''

    numerator = 0
    denom1 = 0
    denom2 = 0
    list_word_1 = s1.split(' ')
    list_word_2 = s2.split(' ')

    for word in list_word_1:
        denom1 += (tf(word, s1) * idf(word, list_sent)) ** 2

    for word in list_word_2:
        tf_s2 = tf(word, s2)
        idf_w = idf(word, list_sent)
        denom2 += (tf_s2 * idf_w) ** 2
        if word in list_word_1:
            numerator += tf(word, s1) * tf_s2 * (idf_w ** 2)
    sim = 0.0
    try:
        sim = numerator / (math.sqrt(denom1) * math.sqrt(denom2))
    except Exception:
        print(s1)
        print(s2)

    return sim


def freq_word(ngram):
    """
    get words has tf on top(30%)
    :param ngram:
    :return:
    """

    if ngram == 1:
        path_data = DATA_REMOVED_STOPWORDS
        output = TOP_TF_UNI
    else:
        path_data = DATA_BIGRAM
        output = TOP_TF_BI

    for filename in os.listdir(path_data):
        arr_tf = []
        with open(path_data + '/' + filename, 'r') as f:
            doc = f.read()
            # remove  '------', '\n' separate documents
            doc_token = doc.replace('\n------\n', ' ').replace('\n', ' ').split(' ')
            print(doc_token)
            len_doc = len(doc_token)
            list_words_count = Counter(doc_token)

            for word, count in list_words_count.items():
                tf_w = count / len_doc
                arr_tf.append((word, tf_w))
        arr_top_tf = sorted(arr_tf, key=lambda x: x[1], reverse=True)
        arr_top_tf_round = []
        for word, score in arr_top_tf[:int(0.3 * len(arr_top_tf))]:
            score = round(score, 5)
            arr_top_tf_round.append(word + ' ' + str(score))

        write_arr_string(arr_top_tf_round, output + '/' + filename)


def get_list_first_sent():
    """
    get list first sentence
    :return:
    """
    with open(FIRST_SENT_DOC, 'r') as f:
        list_first_sent = f.read().split('\n')
    return list_first_sent


def first_rel_doc():
    '''
    compute relevant with first sentence of document
    compute by cosine similarity tfidf
    :return:
    '''

    for filename in os.listdir(DATA_REMOVED_STOPWORDS):
        print("cal", filename)

        similarities = []
        list_sent = []
        with open(DATA_REMOVED_STOPWORDS + '/' + filename, 'r') as fout:
            # read docs, split by '\n------\n' to have list docs separate
            files = fout.read().strip().split('\n------\n')
            for file in files:
                for sent in file.strip().split('\n'):
                    list_sent.append(sent)

            for file in files:
                sub_list_sent = file.strip().split('\n')
                first_sent = sub_list_sent[0]
                similarities.append('1.0' + '::' + first_sent)
                for sent in sub_list_sent[:-1]:
                    simil_value = cos_similarity(first_sent, sent, list_sent)
                    similarities.append(str(simil_value) + '::' + sent)

        write_arr_string(similarities, SENT_SIMILARITIES + '/' + filename)


freq_word(1)
