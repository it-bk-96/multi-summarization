import os
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer
from nltk.util import ngrams

DATA_200_CLUSTERS = 'Data_Non_Token/Documents'
STOPWORDS = 'stopwords.txt'
FIRST_SENT_DOC = 'First_Sentence'
CLUSTERS_TOKEN = 'Data_Token'
DATA_TOKEN_WORDS = 'Data_Token_Words'
DATA_REMOVED_STOPWORDS = 'Data_Remove_Stopwords'
DATA_BIGRAM = 'Data_Bigram'
SPECIAL_CHARACTER = ['(', ')', '[', ']', '"', '”', '“', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# def token_sent():
#     for dir in os.listdir(DATA_200_CLUSTERS):
#
#         subdir = DATA_200_CLUSTERS + '/' + dir
#         cluster = []
#         for filename in os.listdir(subdir):
#             with open(subdir + "/" + filename) as tex:
#                 cluster.append(tex.read().strip())
#
#         write_token_sent(cluster, CLUSTERS_TOKEN + '/' + dir)


def token_sent():
    for dir in os.listdir(DATA_200_CLUSTERS):

        subdir = DATA_200_CLUSTERS + '/' + dir
        cluster = []
        for filename in os.listdir(subdir):
            with open(subdir + "/" + filename) as tex:
                cluster.append(tex.read().strip())

        write_token_sent(cluster, CLUSTERS_TOKEN + '/' + dir)


# def token_word(input, output):
#     for filename in os.listdir(input):
#         clus = []
#         with open(input + "/" + filename) as tex:
#             for line in tex.read().split('\n'):
#                 line_token = ViTokenizer.tokenize(line)
#                 clus.append(line_token)
#         write_token_sent(clus, output + "/" + filename)


def token_word(input, output):
    for filename in os.listdir(input):
        clus = []
        with open(input + "/" + filename) as tex:
            # separate documents
            data = tex.read().split('\n------\n')
            for file in data:
                for line in file.strip().split('\n'):
                    line_token = ViTokenizer.tokenize(line)
                    clus.append(line_token)
                # separate documents by ------
                clus.append('------')
        write_arr_string(clus[:-1], output + "/" + filename)


def to_bigram(token):
    bigrams = list(ngrams(token, 2))
    bigram_data = ''
    for word1, word2 in bigrams:
        bigram_data += word1 + "_" + word2 + ' '
    return bigram_data


def token_bigram():
    for filename in os.listdir(DATA_REMOVED_STOPWORDS):
        clus = []
        with open(DATA_REMOVED_STOPWORDS + "/" + filename) as tex:
            lines = tex.read().split('\n')
            for line in lines:
                bi_line = to_bigram(line.split(" "))
                clus.append(bi_line)
        write_arr_string(clus, DATA_BIGRAM + '/' + filename)


# def get_first_sent():
#
#     for dir in os.listdir(DATA_200_CLUSTERS):
#         list_first_sent = []
#         subdir = DATA_200_CLUSTERS + '/' + dir
#         for filename in os.listdir(subdir):
#             with open(subdir + "/" + filename) as tex:
#                 list_sent = sent_tokenize(tex.read())
#                 list_first_sent.append(list_sent[0])
#
#             # write list first sentences to file
#             f = open(FIRST_SENT_DOC + '/' + dir, 'w')
#             f.write('\n'.join(list_first_sent))


def write_arr_string(arr_str, output):
    f = open(output, 'w')
    f.write('\n'.join(arr_str))


# def write_token_sent(cluster, path_output):
#     clus_tok = []
#     for doc in cluster:
#         list_sent = sent_tokenize(doc)
#         for sent in list_sent:
#             clus_tok.append(sent)
#
#     # write list sentence of a cluster
#     f1 = open(path_output, 'w')
#     f1.write('\n'.join(clus_tok))


def write_token_sent(cluster, path_output):
    clus_tok = []
    for doc in cluster:
        list_sent = sent_tokenize(doc)
        for sent in list_sent:
            clus_tok.append(sent)
        # separate documents by ------
        clus_tok.append('------')
    # write list sentence of a cluster
    f1 = open(path_output, 'w')
    f1.write('\n'.join(clus_tok[:-1]))


def get_stopwords():
    with open(STOPWORDS) as st:
        stopwords = st.read().split('\n')

    return stopwords


# def remove_stopwords(input, output):
#     list_stopwords = get_stopwords()
#     for filename in os.listdir(input):
#         print(filename)
#         clus = []
#         with open(input + "/" + filename) as tex:
#             lines = tex.read().split('\n')
#             for line in lines:
#                 stem_sent_tok = ViTokenizer.tokenize(line.strip()).lower().split()
#                 stem_sent_tok_pro = list(
#                     filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
#                                      and x != '!' and x != '''"''' and x != "''" and x != '-' and x != ')'
#                                      and x != '(' and x != '“' and x != '”' and x != '–' and x != '...'
#                                      and x != '/' and x != ';' and x != '%' and x not in list_stopwords,
#                            stem_sent_tok))
#                 if len(stem_sent_tok_pro) > 2:
#                     clus.append(' '.join(stem_sent_tok_pro))
#         write_token_sent(clus, output + '/' + filename)

def remove_stopwords(path_data, path_output):



def label_data(path_extract, path_data):
    for dir in os.listdir(path_data):
        list_sents =

# remove_stopwords(DATA_TOKEN_WORDS, DATA_REMOVED_STOPWORDS)
token_word(CLUSTERS_TOKEN, DATA_TOKEN_WORDS)
remove_stopwords(DATA_TOKEN_WORDS, DATA_REMOVED_STOPWORDS)
