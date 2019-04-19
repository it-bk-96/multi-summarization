import re
import nltk
from pyvi import ViTokenizer
porter = nltk.PorterStemmer()

SPECICAL_CHARACTER = {'(', ')', '[', ']', ',', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


def text_process_english(sentences):
    new_sentences = []
    original_sent = []

    for item in sentences:
        tmp = re.sub('[<>@~:.;]', '', item)
        tmp = re.sub('-', ' ', tmp)
        tmp = re.sub('[“”]', '"', tmp)
        text_tmp = []
        token_sent = nltk.word_tokenize(tmp)
        token_sent = [porter.stem(word) for word in token_sent]
        for word in token_sent:
            if len(word) != 1 or word in SPECICAL_CHARACTER:
                text_tmp.append(word)

        if len(text_tmp) > 5:
            new_sentences.append(' '.join(list(map(lambda x:x.lower(),text_tmp))).strip())
            original_sent.append(tmp)

    return new_sentences, original_sent


def split_sentences(file_name):
    try:
        with open(file_name, 'r') as file:
            text_system = file.read().strip()

        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = sentence_token.tokenize(text_system)
        preprocess_sents, original_sents = text_process_english(sentences)
        return preprocess_sents, original_sents

    except Exception as e:
        print(e)


def get_all_sentences(file_system, file_reference):
    sentences_system = []
    sentences_origin_system = []
    for item in file_system:
        sent_system, sent_original_system = split_sentences(item)

        sentences_system.append((item, sent_system))
        sentences_origin_system.append(sent_original_system)

    sentences_reference = []
    for item in file_reference:
        with open(item, 'r') as file:
            sentences_ref, oriaaaa = text_process_english(nltk.sent_tokenize(file.read()))
            sentences_reference.append('. '.join(sentences_ref))

    return sentences_origin_system, sentences_system, sentences_reference
