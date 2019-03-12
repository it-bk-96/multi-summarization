# import pickle
# import numpy as np
# from sklearn.decomposition import NMF
#
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# model = NMF(n_components=2, init='random', random_state=0)
# W = np.array(model.fit_transform(X))
# H = np.array(model.components_)
# print(W)
# print(H)
# print(W.dot(H))
#
# with open("/home/giangvu/Desktop/multi-summarization/models/w2v.model", 'rb') as f:
#     model_w2v = pickle.load(f)
# word = "em"
# w_vec = np.array(model_w2v[word])
# print(len(w_vec))
# print(model_w2v)
# print(w_vec)

from bert import extract_features

vocab_file = '/home/giangvu/Desktop/multi-summarization/bert/multi_cased_L-12_H-768_A-12/vocab.txt'
bert_config_file = '/home/giangvu/Desktop/multi-summarization/bert/multi_cased_L-12_H-768_A-12/bert_config.json'
layers = "-1, -2, -3, -4"
init_checkpoint = '/home/giangvu/Desktop/multi-summarization/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
max_seq_length = 128
batch_size = 8

sentences = ["Anh hơn ẻm khổng", "Tôi không yêu em"]


# vector_sentences = extract_features.getFeature(sentences, bert_config_file, vocab_file, init_checkpoint, layers, max_seq_length, batch_size)

tokenization = extract_features.tokenization
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
print(sentences[0])
print(tokenizer.tokenize(sentences[0]))
print(sentences[1])
print(tokenizer.tokenize(sentences[1]))

# print(vector_sentences[0])
# print(vector_sentences[1])
