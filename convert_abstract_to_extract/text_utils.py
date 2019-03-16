from pyvi import ViTokenizer
import nltk
import re

SPECICAL_CHARACTER = {'(', ')', '[', ']', '”', '“', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
QUOTE = {'(', ')', '[', ']', '”', '“', '*'}

def text_process(sentences):
    new_sentences = []

    for item in sentences:
        tmp = re.sub('[<>@!\-~:.;*]', '', item)
        text_tmp = ""
        token_sent = ViTokenizer.tokenize(tmp).lower()
        for word in token_sent.split(' '):
            if len(word) != 1 or word in SPECICAL_CHARACTER:
                text_tmp += word + " "
        new_sentences.append(text_tmp[:-1].strip())

    return new_sentences



def split_sentences(file_name):
    try:
        with open(file_name, 'r') as file:
            text_system = file.read()

        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
        tmp = sentence_token.tokenize(text_system)
        preprocess_sents = text_process(tmp)
        sentences = []
        for item in preprocess_sents:
            if "…" in item:
                b = item.split("…")
                for i in b:
                    sentences.append(i)
            else:
                sentences.append(item)

        return sentences

    except Exception:
        print(file_name)


# def get_all_sentences(file_system, file_reference):
#     sentences_origin_system = []
#     for item in file_system:
#         sentences_origin_system.append((item, split_sentences(item)))
#
#     sentences_system = []  # be token words
#     for filename, list_sent_origin in sentences_origin_system:
#         sent_in_clus = []
#         for item in list_sent_origin:
#             sent_in_clus.append(ViTokenizer.tokenize(item))
#         sentences_system.append((filename, sent_in_clus))
#
#
#     sentences_origin_reference = []
#     for item in file_reference:
#         with open(item, 'r') as file:
#             sentences_origin_reference.append(file.read())
#
#     sentences_reference = []
#     for item in sentences_origin_reference:
#         sentences_reference.append(ViTokenizer.tokenize(item))
#
#     return sentences_system, sentences_reference



def get_all_sentences(file_system, file_reference):
    sentences_origin_system = []
    for item in file_system:
        sentences_origin_system.append((item, split_sentences(item)))



    sentences_reference = []
    for item in file_reference:
        with open(item, 'r') as file:
            sentences_ref = text_process(nltk.sent_tokenize(file.read()))
            sentences_reference.append('. '.join(sentences_ref))

    return sentences_origin_system, sentences_reference

