# import re
# import nltk
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.lancaster import LancasterStemmer
#
# SPECICAL_CHARACTER = {'"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
# QUOTE = {'"'}
#
#
# def read_file_text(path):
#     f = open(path, 'r')
#     return f.read()
#
# def extract_feature4(sentence):
#     words = []
#     print(sentence.split(' '))
#     for item in sentence.split(' '):
#         if item in QUOTE:
#             words.append(item)
#
#     return len(words)
#
# def text_process(sentences, stop_words):
#     match = {
#         ' not': 'n\'t',
#         '': '(\'s|\'ll|\'re|\'d|\'ve)',
#         ' ': '[^a-zA-Z0-9"]',
#         ' " ': '"'
#     }
#
#     for key in match:
#         sentences = [sent[0:2] + re.sub(match[key], key, sent[2:]) for sent in sentences]
#
#     new_sentences = []
#
#     for item in sentences:
#         text_tmp = ""
#         for word in item.lower().split(' '):
#             l = len(word)
#             if (word not in stop_words) and l != 0:
#                 text_tmp += word + " "
#         new_sentences.append(text_tmp)
#     return new_sentences
#
# def text_process_all(sentences, stop_words):
#     new_sentences = []
#     lematizer = WordNetLemmatizer()
#     stemmer = LancasterStemmer()
#     match = {
#         ' not': 'n\'t',
#         '': '(\'s|\'ll|\'re|\'d|\'ve)',
#         ' ': '[^a-zA-Z0-9]'
#     }
#
#     sentences_nolabel = [sent[2:] for sent in sentences]
#     for key in match:
#         sentences_nolabel = [re.sub(match[key], key, sent) for sent in sentences_nolabel]
#
#     for item in sentences_nolabel:
#         tmp = item.strip().lower()  # remove label from data => [2:]
#         text_tmp = []
#         for word in tmp.split(' '):
#             l = len(word)
#             if (word not in stop_words) and l != 0:
#                 word = stemmer.stem(lematizer.lemmatize(word))
#                 text_tmp.append(word)
#
#
#         new_sentences.append(' '.join(text_tmp))
#
#     return new_sentences
#
#
# def get_all_idf(documents):
#     words = {}
#
#     for item in documents:
#         for word in nltk.word_tokenize(item):
#             if word not in QUOTE and word not in words:
#                 words[word] = 0.0
#
#     # for item in words:
#     #     words[item] = idf(item, documents)
#
#     return words
#
# review = read_file_text("/home/hieupd/PycharmProjects/multi_summari_svm_english/data_labels/train/cluster_01/APW19990612.0141").split('\n')
#
# stops = read_file_text("./stopwords.txt").strip().split("\n")
#
# a = text_process(review, stops)

