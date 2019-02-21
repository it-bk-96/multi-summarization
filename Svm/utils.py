from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer
import os
import math
from data_interact import write_arr_string
from collections import Counter

TOP_TF_UNI = 'Top_TF_Uni'
TOP_TF_BI = 'Top_TF_Bi'
DATA_200_CLUSTERS = 'Data_Non_Token/Documents'
STOPWORDS = 'stopwords_2.txt'
FIRST_SENT_DOC = 'first_sentence'
CLUSTERS_TOKEN = 'Data_Token'
DATA_TOKEN_WORDS = 'Data_Token_Words'
DATA_REMOVED_STOPWORDS = 'Data_Remove_Stopwords_2'
DATA_BIGRAM = 'Data_Bigram'
SENT_SIMILARITIES = 'Sent_Similarities'
TFIDF_UNI = 'Tfidf_Uni'




def tf(word, doc):
    count_w = doc.count(word)
    count_word_doc = len(ViTokenizer.tokenize(doc))

    return count_w / count_word_doc

def count_num_doc_contain(w,listdoc):
    num = 0
    for doc in listdoc:
        if w in doc.split(' '):
            num += 1
    return num

def idf(word, list_doc):
    num_doc = len(list_doc)
    return math.log(num_doc / count_num_doc_contain(word, list_doc))

def tfidf(word, doc, list_doc):
    return tf(word, doc) / idf(word, list_doc)

# get all document to calculate tfidf
def get_all_doc(path):
    all_docs = []
    for filename in os.listdir(DATA_REMOVED_STOPWORDS):
        with open(DATA_REMOVED_STOPWORDS + '/' + filename) as f:
            all_docs.append(f.read().replace('\n', ''))

    return all_docs

def cal_tfidf_alldoc():
    all_docs = get_all_doc(DATA_REMOVED_STOPWORDS)

    for filename in os.listdir(DATA_REMOVED_STOPWORDS):
        arr_tfidf = []
        with open(DATA_REMOVED_STOPWORDS + '/' + filename) as f:
            doc = f.read().replace('\n', '')
            list_words = set(doc.split(' '))
            for w in list_words:
                arr_tfidf.append((w, tfidf(w, doc, all_docs)))
        arr_tfidf_sorted = sorted(arr_tfidf, key= lambda x: x[1])
        arr_tfidf_top = []

        for word, score in arr_tfidf_sorted[:round(0.3*len(arr_tfidf))]:
            score = round(score,5)
            arr_tfidf_top.append(w + ':' + str(score))

        write_arr_string(arr_tfidf_top, TFIDF_UNI + '/' + filename)





def cos_similarity(s1, s2):
    pass


def sig_term_uni():
    pass


def sig_term_bi():
    pass


def freq_word(ngram):
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
            doc_token = doc.replace('\n', '').split(' ')
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
    with open(FIRST_SENT_DOC, 'w') as f:
        list_first_sent = f.read().split('\n')
    return list_first_sent


# compute by cosine similarity or count word in common
def first_rel_doc():
    list_first_sent = get_list_first_sent()
    for filename in os.listdir(DATA_REMOVED_STOPWORDS):
        similarities = []
        with open(DATA_REMOVED_STOPWORDS + '/' + filename, 'r') as fout:
            for line in fout.read().split('\n'):
                first_sent =''
                if line in list_first_sent:
                    first_sent = line
                simil_value = cos_similarity(first_sent, line)
                similarities.append(line + ' :' + str(simil_value))

        write_arr_string(similarities, SENT_SIMILARITIES)

cal_tfidf_alldoc()
