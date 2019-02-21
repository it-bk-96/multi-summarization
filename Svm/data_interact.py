import os
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer
from nltk.util import ngrams



DATA_200_CLUSTERS = 'Data_Non_Token/Documents'
STOPWORDS = 'stopwords_2.txt'
FIRST_SENT_DOC = 'first_sentence'
CLUSTERS_TOKEN = 'Data_Token'
DATA_TOKEN_WORDS = 'Data_Token_Words'
DATA_REMOVED_STOPWORDS = 'Data_Remove_Stopwords_2'
DATA_BIGRAM = 'Data_Bigram'


def token_sent():
    for dir in os.listdir(DATA_200_CLUSTERS):

        subdir = DATA_200_CLUSTERS + '/' + dir
        cluster = []
        for filename in os.listdir(subdir):
            with open(subdir + "/" + filename) as tex:
                cluster.append(tex.read())

        write_token_sent(cluster, CLUSTERS_TOKEN + '/' + dir)


def token_word():
    for filename in os.listdir(CLUSTERS_TOKEN):
        clus = []
        with open(CLUSTERS_TOKEN + "/" + filename) as tex:
            for line in tex.read().split('\n'):
                line_token = ViTokenizer.tokenize(line)
                clus.append(line_token)
        write_token_sent(clus, DATA_TOKEN_WORDS + "/" + filename)

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



def get_first_sent():
    list_first_sent = []
    for dir in os.listdir(DATA_200_CLUSTERS):
        subdir = DATA_200_CLUSTERS + '/' + dir
        for filename in os.listdir(subdir):
            with open(subdir + "/" + filename) as tex:
                list_sent = sent_tokenize(tex.read())
                list_first_sent.append(list_sent[0])

    # write list first sentences to file
    f = open(FIRST_SENT_DOC, 'w')
    f.write('\n'.join(list_first_sent))

def write_arr_string(arr_str, output):
    f = open(output, 'w')
    f.write('\n'.join(arr_str))

def write_token_sent(cluster, path_output):
    clus_tok = []
    for doc in cluster:
        list_sent = sent_tokenize(doc)
        for sent in list_sent:
            clus_tok.append(sent)

    # write list sentence of a cluster
    f1 = open(path_output, 'w')
    f1.write('\n'.join(clus_tok))


def get_stopwords():
    with open(STOPWORDS) as st:
        stopwords = st.read().split('\n')

    return stopwords

def remove_stopwords():
    list_stopwords = get_stopwords()
    for filename in os.listdir(DATA_TOKEN_WORDS):
        clus = []
        with open(DATA_TOKEN_WORDS + "/" + filename) as tex:
            lines = tex.read().split('\n')
            for line in lines:
                stem_sent_tok = ViTokenizer.tokenize(line.strip()).lower().split()
                stem_sent_tok_pro = list(
                    filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                                     and x != '!' and x != '''"''' and x != "''" and x != '-' and x != ')'
                                     and x != '(' and x != '“' and x != '”' and x != '–' and x != '...'
                                     and x != '/' and x != ';' and x != '%' and x not in list_stopwords,
                           stem_sent_tok))
                if len(stem_sent_tok_pro) > 2:
                    clus.append(' '.join(stem_sent_tok_pro))
        write_token_sent(clus, DATA_REMOVED_STOPWORDS + '/' + filename)


#token_bigram()