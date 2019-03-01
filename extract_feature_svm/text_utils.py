from pyvi import ViTokenizer
import nltk
import operator
import math

SPECICAL_CHARACTER = {'(', ')', '[', ']', '”', '“', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
QUOTE = {'(', ')', '[', ']', '”', '“', '*'}

def split_sentences(file_name):
    with open(file_name, 'r') as file:
        text_system = file.read()

    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    tmp = sentence_token.tokenize(text_system)

    sentences = []
    for item in tmp:
        if "…" in item:
            b = item.split("…")
            for i in b:
                sentences.append(i)
        else:
            sentences.append(item)

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

def text_process(sentences, stop_words):
    new_sentences = []

    for item in sentences:
        tmp = ViTokenizer.tokenize(item)
        text_tmp = ""
        for word in tmp.split(' '):
            if (word not in stop_words) and (len(word) != 1 or word in SPECICAL_CHARACTER):
                text_tmp += word + " "

        new_sentences.append(text_tmp[:-1])

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

def tf(word, document):
    dict_words = get_dict_words_from_doc(document)
    count_words = 0

    for item in dict_words:
        count_words += dict_words[item]

    return dict_words[word]*1.0/count_words

def get_word_freq(word, document):
    dict_words = get_dict_words_from_doc(document)

    if word not in dict_words:
        return 0

    return dict_words[word]

def idf(word, documents):
    N = len(documents)

    contain_word = 0
    for item in documents:

        if word in item:
            contain_word += 1

    return math.log(1.0*N/contain_word)

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
        words[item] = tf(item, document) * all_idf[item]

    number_centroid_uni = int(0.3*len(words))

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
        sentences = split_sentences(item)
        sentences = text_process(sentences, stop_words)

        documents.append(get_doc_from_sentences(sentences))

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
            bi_document.append(words[item-1] + "__" + words[item])

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
        denom1 += (tf_arr[word] * idf[word]) ** 2

    for word in list_word_s2:

        denom2 += (tf_arr[word] * idf[word]) ** 2
        if word in list_word_s1:
            numerator += (tf_arr[word] * idf[word]) ** 2
    sim = 0.0
    try:
        sim = numerator / (math.sqrt(denom1) * math.sqrt(denom2))
    except Exception:
        pass

    return sim

if __name__ == "__main__":
    sentences = split_sentences("document1.txt")
    sentences2 = split_sentences("document2.txt")
    # # print (sentences)
    #
    # # print (sentences)
    #
    stop_words = read_stopwords("stop_words.txt")
    # # print (stop_words)
    #
    sentences = text_process(sentences, stop_words)
    doc = get_doc_from_sentences(sentences)

    sentences2 = text_process(sentences2, stop_words)
    doc2 = get_doc_from_sentences(sentences2)

    # print (tf("Người", doc))
    a = get_freq_word_uni(doc)

    # print (a)
    # a = convert_uni_to_bi([doc])
    #
    # print (a)

    all_idf = get_all_idf([doc, doc2])
    print (all_idf)

    print (cos_similarity(sentences[1], all_idf, sentences))

    # print (all_idf)
    #
    # a = get_centroid_uni(doc, all_idf)
    #
    # print (a)
    # print (get_all_word([sentences, sentences]))
    # for item in sentences:
    #     print (item)
    # print ("----------------------------------")
    # contain = get_sentence_first_paragraph("document1.txt", stop_words)
    #
    # for item in sentences:
    #     if item in contain:
    #         print (1)
    #     else:
    #
    #         print (0)