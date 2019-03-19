import nltk
import re

SPECICAL_CHARACTER = {'(', ')', '[', ']', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
QUOTE = {'(', ')', '[', ']', '"'}

def text_process_english(sentences):
    new_sentences = []

    for item in sentences:
        tmp = re.sub('[<>@!\-~:.;*]', '', item)
        text_tmp = []
        token_sent = tmp.strip().lower()
        for word in token_sent.split(' '):
            if len(word) != 1 or word in SPECICAL_CHARACTER:
                text_tmp.append(word)
        new_sentences.append(' '.join(text_tmp))

    return new_sentences



def split_sentences(file_name):
    try:
        with open(file_name, 'r') as file:
            text_system = file.read()

        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
        tmp = sentence_token.tokenize(text_system)
        preprocess_sents = text_process_english(tmp)
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



def get_all_sentences(file_system, file_reference):
    sentences_origin_system = []
    for item in file_system:
        sentences_origin_system.append((item, split_sentences(item)))



    sentences_reference = []
    for item in file_reference:
        with open(item, 'r') as file:
            sentences_ref = text_process_english(nltk.sent_tokenize(file.read()))
            sentences_reference.append('. '.join(sentences_ref))

    return sentences_origin_system, sentences_reference

