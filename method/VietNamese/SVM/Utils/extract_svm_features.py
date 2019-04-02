import os
from definitions import ROOT_DIR
from method.VietNamese.SVM.Utils import text_utils
# from method.VietNamese.SVM.Scores import nmf_summarizer
# from method.VietNamese.SVM.Scores import LexRank
# from method.VietNamese.SVM.Scores import TextRank
# import json

SPECICAL_CHARACTER = {'(', ')', '[', ']', '”', '“', '*'}
SVM_FEATURES = ROOT_DIR + '/method/VietNamese/SVM/Data/svm_features'


class FeatureSvm(object):
    def __init__(self, file_name, file_stopwords):
        self.file_name = file_name
        self.stop_words = text_utils.read_stopwords(file_stopwords)

    def get_sentences(self):
        sentences = text_utils.split_sentences(self.file_name)
        return text_utils.text_process(sentences, self.stop_words)

    # 1/Position
    def extract_feature1(self, sentence, sentences):
        return 1.0 / (sentences.index(sentence) + 1)

    # Doc_First
    def extract_feature2(self, sentence):
        contain_first_paragraph = text_utils.get_sentence_first_paragraph(self.file_name, self.stop_words)

        if sentence in contain_first_paragraph:
            return 1
        else:
            return 0

    # Length
    def extract_feature3(self, sentence):
        words = []
        for item in sentence.split(' '):
            if item not in SPECICAL_CHARACTER:
                words.append(item)

        return len(words)

    # Quote
    def extract_feature4(self, sentence):
        words = []
        for item in sentence.split(' '):
            if item in SPECICAL_CHARACTER:
                words.append(item)

        return len(words)

    # FreqWord_Uni, FreqWord_Bi
    def extract_feature7_8(self, sentence, document):
        freq_words = text_utils.get_freq_word_uni(document)

        feature = 0.0
        for item in sentence.split(' '):
            if item not in freq_words:
                continue
            else:

                feature += freq_words[item]

        return feature

    # Centroid_Uni, Centroid_Bi
    def extract_feature5_6(self, sentence, document, all_idf):

        centroid_uni = text_utils.get_centroid_uni(document, all_idf)

        feature = 0.0
        for item in sentence.split(' '):
            if item not in centroid_uni:
                continue
            else:
                feature += centroid_uni[item]

        return feature

    # FirstRel_Doc
    def extract_feature9(self, sentence, all_idf, sentences):
        return text_utils.cos_similarity(sentence, all_idf, sentences)

    def get_all_feature_from_sentence(self, sentence, sentences, all_idf, bi_sentence, bi_all_idf):
        document = text_utils.get_doc_from_sentences(sentences)
        bi_document = text_utils.convert_uni_to_bi([document])[0]


        feature1 = self.extract_feature1(sentence, sentences)
        feature2 = self.extract_feature2(sentence)
        feature3 = self.extract_feature3(sentence)
        feature4 = self.extract_feature4(sentence)
        feature5 = self.extract_feature5_6(sentence, document, all_idf)
        feature6 = self.extract_feature5_6(bi_sentence, bi_document, bi_all_idf)
        feature7 = self.extract_feature7_8(sentence, document)
        feature8 = self.extract_feature7_8(bi_sentence, bi_document)
        feature9 = self.extract_feature9(sentence, all_idf, sentences)

        return [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]

    def get_all_feature_from_doc(self, sentences, all_idf, bi_all_idf):

        features = []
        for item in sentences:
            bi_sentence = text_utils.convert_uni_to_bi([item])[0]

            feature = self.get_all_feature_from_sentence(item, sentences, all_idf, bi_sentence, bi_all_idf)
            features.append(feature)

        return features

if __name__ == "__main__":

    path_stopword = ROOT_DIR + '/vietnamese-stopwords.txt'
    stop_words = text_utils.read_stopwords(path_stopword)

    root = ROOT_DIR + "/method/VietNamese/SVM/Data/data_labels"

    # compute idf
    list_document_paths = []

    i = 0
    for clus in os.listdir(root + '/train'):
        # i += 1
        # if i > 20:
        #     break
        path_clus = root + '/train/' + clus
        for filename in os.listdir(path_clus):
            list_document_paths.append(path_clus + '/' + filename)


    # read all doc and preprocess data
    documents = text_utils.read_all_documents(list_document_paths, stop_words)

    # bag = []
    # for i in documents:
    #     bag += i.split(' ')

    bi_documents = text_utils.convert_uni_to_bi(documents)
    all_idf = text_utils.get_all_idf(documents)
    text_utils.save_idf(all_idf, 'all_idf.json')

    bi_all_idf = text_utils.get_all_idf(bi_documents)
    text_utils.save_idf(bi_all_idf, 'all_bi_idf.json')
    # all_idf = text_utils.read_json_file('all_idf.json')
    # bi_all_idf = text_utils.read_json_file('all_bi_idf.json')


    a = 0

    for dir in os.listdir(root):
        path_dir = root + '/' + dir
        for clus in os.listdir(path_dir):
            # a += 1
            # if a > 20:
            #     break
            print(clus)
            path_clus = path_dir + '/' + clus
            sents_of_clus = []
            all_features = []
            arr_features_svm = []
            arr_labels = []
            for filename in os.listdir(path_clus):
                # initial extract feature

                extract_feature = FeatureSvm(path_clus + '/' + filename,
                                             path_stopword)

                sentences = extract_feature.get_sentences() # be lowered

                # remove short sentences, return stem sentences and origin
                sentences_stem, sentences_origin = text_utils.remove_short_sents(sentences)

                # get list label of each file
                labels, sents_nolabel = text_utils.separate_label_sent(sentences_origin)
                arr_labels += labels

                # add to list sents of cluster
                sents_of_clus += sentences_stem

                arr_features_svm += extract_feature.get_all_feature_from_doc(sents_nolabel, all_idf, bi_all_idf)
            arr_svm_normal = text_utils.normalize(arr_features_svm)
            # arr_scores_NMF = nmf_summarizer.getNMF(sents_of_clus)
            # arr_scores_Lexrank = LexRank.getLexRank(sents_of_clus)
            # #arr_scores_textrank = TextRank.getTextRank(sents_of_clus)

            dir_output = SVM_FEATURES + '/' + dir
            text_utils.prepare_data_svm(arr_labels, arr_svm_normal, dir_output + '/' + clus)
