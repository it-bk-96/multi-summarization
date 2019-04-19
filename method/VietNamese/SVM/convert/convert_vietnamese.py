import os
import json
from definitions import ROOT_DIR
from rouge import pyrouge_vn_200_cluster
from method.VietNamese.SVM.convert import text_utils_vietnamese

DIR_PATH = ROOT_DIR + '/Data/VietNamese'


class ConvertExtract(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def convert_extract(self, list_documents, list_human, path_save):
        sentences_origin_system, sentences_system, sentences_reference = \
            text_utils_vietnamese.get_all_sentences(list_documents, list_human)

        old_rouge = 0
        rouge = 0  # initial rouge score
        old_index = -1

        sentences = []  # arr sentence is choosed
        sentences_label = []
        arr_indexes = []
        all_sentences = []
        all_sentences_origin = []

        for filename, list_sents in sentences_system:
            all_sentences += list_sents
            # with each round with add new sentence

        for list_sents in sentences_origin_system:
            all_sentences_origin += list_sents

        while (0 == 0):
            i = 0

            for sent in all_sentences:
                tmp = ""
                for s in sentences:
                    tmp += s

                tmp += sent

                # Use rouge 1
                tmp_rouge_f1 = pyrouge_vn_200_cluster.rouge_1(tmp, sentences_reference, self.alpha)

                if tmp_rouge_f1 > rouge:  # if has change score
                    rouge = tmp_rouge_f1
                    old_index = i

                i += 1

            if rouge == old_rouge:
                break
            else:
                arr_indexes.append(old_index)
                old_rouge = rouge
                sentences.append(all_sentences[old_index])
                old_index = -1

        for i in range(len(all_sentences)):
            if i in arr_indexes:
                sentences_label.append({
                    'label': '1',
                    'tokenized': all_sentences[i],
                    'origin': all_sentences_origin[i],
                })

            else:
                sentences_label.append({
                    'label': '0',
                    'tokenized': all_sentences[i],
                    'origin': all_sentences_origin[i],
                })
        length = 0
        sentences_label_clus = []
        for clus, list_sentences in sentences_system:
            l = len(list_sentences)
            number_sents = len(list_sentences)
            sentences_label_clus.append((clus, sentences_label[length:number_sents + length]))
            length += l
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        for path_file, data in sentences_label_clus:
            with open(path_save + '/' + path_file.split('/')[-1], 'w') as file:
                json.dump(data, file)
        return rouge


if __name__ == "__main__":
    convert_extract = ConvertExtract()

    list_clusters = []
    list_human_refs = []
    dir_documents = DIR_PATH + '/Documents'
    for clus in os.listdir(dir_documents):
        list_clusters.append(dir_documents + '/' + clus)
        id_cluster = clus.split("_").pop()
        list_human_refs.append(
            [DIR_PATH + '/Human_Summaries/' + clus + '.ref1.txt', DIR_PATH + '/Human_Summaries/' + clus + '.ref2.txt'])

    rouges = []
    for i in range(len(list_clusters)):
        list_docs_in_clus = []
        list_human_ref = []
        name_cluster = list_clusters[i]
        for filename in os.listdir(name_cluster):
            list_docs_in_clus.append(name_cluster + '/' + filename)
        name_clus = name_cluster.split('/')[-1]

        for ref in list_human_refs[i]:
            list_human_ref.append(ref)
        rouges.append(convert_extract.convert_extract(list_docs_in_clus, list_human_ref,
                                                      ROOT_DIR + "/method/VietNamese/SVM/Data/converted/" + name_clus))

    print(sum(rouges) / len((rouges)))
