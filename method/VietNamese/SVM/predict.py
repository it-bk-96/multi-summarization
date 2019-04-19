import os
import json
import numpy as np
from joblib import load
from pyvi import ViTokenizer
from method.VietNamese.SVM.Utils import text_utils
from method.VietNamese.SVM import mmr_selection
from definitions import ROOT_DIR
np.set_printoptions(threshold=np.inf)


def get_doc_id(index, leng_docs):
    sum_len = 0
    id_doc = 0
    for i in leng_docs:
        if index < sum_len + leng_docs[i]:
            id_doc = i
            break
        else:
            sum_len += length_docs[i]
    return id_doc


def evaluate_data(X_test):
    # predict document input

    reload = load(ROOT_DIR + "/method/VietNamese/SVM/model")
    predict = reload.predict_proba(X_test)
    sort_predict = sorted(enumerate(predict), key=lambda x: list(x[1])[0], reverse=True)
    return sort_predict

if __name__ == '__main__':

    path_data = ROOT_DIR + '/method/VietNamese/SVM/Data/converted/test'
    path_features = ROOT_DIR + '/method/VietNamese/SVM/Data/svm_features'
    f_test = []

    for clus in os.listdir(path_features + '/' + 'test'):
        t = open(path_features + '/test/' + clus, 'r')
        f_test.append((clus, t.read().split('\n')))
        t.close()

    idf = text_utils.read_json_file(ROOT_DIR + '/method/VietNamese/SVM/Utils/all_idf.json')

    for name_clus, features in f_test:
        print('clus', name_clus)
        X_test = []
        for fea in features:
            X_test.append(list(map(float, fea[2:].split(' ')))) # [:9]
        prob_predic = evaluate_data(np.array(X_test))

        human_path = ROOT_DIR + "/Data/VietNamese/Human_Summaries/"
        file_human_1 = human_path + name_clus + ".ref1.txt"
        text_1_token = open(file_human_1, 'r').read()
        text_1_token = ViTokenizer.tokenize(text_1_token.strip()).lower().split()

        file_human_2 = human_path + name_clus + ".ref2.txt"
        text_2_token = open(file_human_2, 'r').read()
        text_2_token = ViTokenizer.tokenize(text_2_token.strip()).lower().split()
        length_summary = (len(text_1_token) + len(text_2_token)) // 2

        docs = {}
        docs_process = {}
        i = 0
        for filename in os.listdir(path_data + '/' + name_clus):
            with open(path_data + '/' + name_clus + '/' + filename) as json_file:
                docs[i] = json.load(json_file)
                docs_process[i] = ". ".join([d['tokenized']for d in docs[i]])
            i += 1

        arr_all_sents = []
        arr_all_sents_origin = []
        length_docs = {}
        for x in range(len(docs)):
            length_docs[x] = len(docs[x])
            arr_all_sents += [doc['tokenized'] for doc in docs[x]]
            arr_all_sents_origin += [doc['origin'] for doc in docs[x]]

        sen_origin = []
        for s in prob_predic:
            ele = {}
            index = int(s[0])
            ele['doc'] = get_doc_id(index, length_docs)
            ele['value'] = arr_all_sents[index]
            ele['origin'] = arr_all_sents_origin[index]
            ele['score_svm'] = list(s[1])[0]
            sen_origin.append(ele)


        print(len(sen_origin))
        summari = mmr_selection.make_summary(sen_origin, length_summary, 0.5, idf, docs_process)
        summari_sents = [sent['origin'] for sent in summari]
        f = open(ROOT_DIR + '/Data/VietNamese/SVM_results/' + name_clus, 'w')
        f.write(' '.join(summari_sents))

    print(mmr_selection.sys_term, mmr_selection.hu_term)

