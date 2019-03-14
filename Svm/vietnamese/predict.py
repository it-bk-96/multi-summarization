from joblib import load
import text_utils
import numpy as np
import os
from pyvi import ViTokenizer
import mmr_selection

np.set_printoptions(threshold=np.inf)


#
# def sentenceSim(sentence1, sentence2, IDF_w):
# 	numerator = 0
# 	denominator = 0
#
# 	for word in sentence2.getPreProWords():
# 		numerator += sentence1.getWordFreq().get(word, 0) * sentence2.getWordFreq().get(word, 0) * IDF_w.get(word,
# 		                                                                                                     0) ** 2
#
# 	for word in sentence1.getPreProWords():
# 		denominator += (sentence1.getWordFreq().get(word, 0) * IDF_w.get(word, 0)) ** 2
#
# 	# check for divide by zero cases and return back minimal similarity
# 	try:
# 		return numerator / math.sqrt(denominator)
# 	except ZeroDivisionError:
# 		return float("-inf")
#
#
# def MMRScore(Si, query, Sj, lambta, IDF):
# 	Sim1 = sentenceSim(Si, query, IDF)
# 	l_expr = lambta * Sim1
# 	value = [float("-inf")]
#
# 	for sent in Sj:
# 		Sim2 = sentenceSim(Si, sent, IDF)
# 		value.append(Sim2)
#
# 	r_expr = (1 - lambta) * max(value)
# 	MMR_SCORE = l_expr - r_expr
#
# 	return MMR_SCORE

def get_doc_id(index, leng_docs):
    '''

    :param index: index of sent in array all sentences
    :param leng_docs: is a dict has num sent of each doc
    :return:
    '''
    sum_len = 0
    id = 0
    for i in leng_docs:
        if index < sum_len + leng_docs[i]:
            id = i
            break
        else:
            sum_len += length_docs[i]

    return id




def evaluate_data(X_test):
    # predict document input

    reload = load("xgboost_model")
    predict = reload.predict_proba(X_test)
    sort_predict = sorted(enumerate(predict), key=lambda x: list(x[1])[0], reverse=True)
    return sort_predict

if __name__ == '__main__':


    # X_train, Y_train, X_test, Y_test = text_utils.convert_features_svm('/home/hieupd/PycharmProjects/multi_summari_svm/svm_features')
    path_data = '/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/data_labels/test'
    path_features = '/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/svm_features'
    f_test = []

    for clus in os.listdir(path_features + '/' + 'test'):
        t = open(path_features + '/test/' + clus, 'r')
        f_test.append((clus, t.read().split('\n')))
        t.close()

    idf = text_utils.read_json_file('all_idf.json')

    for name_clus, features in f_test:
        print('clus', name_clus)
        X_test = []
        Y_test = []
        for fea in features:
            X_test.append(list(map(float, fea[2:].split(' ')))) # [:9]
            Y_test.append(int(fea[0]))
        prob_predic = evaluate_data(np.array(X_test))

        human_path = "/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/human_test/"
        file_human_1 = human_path + name_clus + '/' + name_clus + ".ref1.txt"
        file_human_2 = human_path + name_clus + '/' + name_clus + ".ref2.txt"
        text_1 = open(file_human_1, 'r').read()
        text_2 = open(file_human_2, 'r').read()
        text_1_token = ViTokenizer.tokenize(text_1)
        text_2_token = ViTokenizer.tokenize(text_2)
        length_summary = int((len(text_1_token.split()) + len(text_1_token.split())) / 2)

        docs = {}
        i = 0
        for filename in os.listdir(path_data + '/' + name_clus):
            f = open(path_data + '/' + name_clus + '/' + filename, 'r')
            docs[i] = f.read()
            i += 1

        arr_all_sents = []
        length_docs = {}
        for x in range(len(docs)):
            sents = docs[x].split('\n')
            length_docs[x] = len(sents)
            arr_all_sents += sents

        sen_origin = []
        for s in prob_predic:
            ele = {}
            index = int(s[0])
            ele['doc'] = get_doc_id(index, length_docs)
            ele['value'] = arr_all_sents[index][2:]
            ele['score_svm'] = list(s[1])[0]
            sen_origin.append(ele)


        print(len(sen_origin))
        summari = mmr_selection.make_summary(sen_origin, length_summary, 0.85, idf, docs)

        summari_sents = [sent['value'] for sent in summari]

        # results = []
        # len_res = 0
        # for i in prob_predic:
        #     index = int(i[0])
        #
        #     if len_res <= length_summary - 5:
        #         results.append(arr_all_sents[index][2:])
        #         len_res += len(arr_all_sents[index][1:].split())

        f = open('/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/results/' + name_clus, 'w')
        f.write(' '.join(summari_sents))

