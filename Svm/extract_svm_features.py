import text_utils
import os
import nmf_summarizer
import LexRank
import TextRank

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"
SPECICAL_CHARACTER = {'(', ')', '[', ']', '”', '“', '*'}


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
        # print (1111111111111111)
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

    path_stopword = '/home/hieupd/PycharmProjects/multi_summari_svm/stopwords.txt'
    stop_words = text_utils.read_stopwords(path_stopword)

    root = "/home/hieupd/PycharmProjects/multi_summari_svm/data_labels"

    # compute idf
    list_document_paths = []
    i = 0
    for clus in os.listdir(root):
        i += 1
        if i == 20:
            break
        path_clus = root + '/' + clus
        for filename in os.listdir(path_clus):
            list_document_paths.append(path_clus + '/' + filename)

    # read all doc and preprocess data
    documents = text_utils.read_all_documents(list_document_paths, stop_words)

    bi_documents = text_utils.convert_uni_to_bi(documents)
    all_idf = text_utils.get_all_idf(documents)
    # print (all_idf)
    bi_all_idf = text_utils.get_all_idf(bi_documents)

    for clus in os.listdir(root):
        path_clus = root + '/' + clus
        sents_of_clus = []
        all_features = []
        arr_vectors = []
        arr_labels = []
        for filename in os.listdir(path_clus):
            # initial extract feature
            extract_feature = FeatureSvm(path_clus + '/' + filename,
                                         path_stopword)

            sentences = extract_feature.get_sentences()

            # remove short sentences, return stem sentences and origin
            sentences_stem, sentences_origin = text_utils.remove_short_sents(sentences)
            print(sentences)
            print("============")
            print(sentences_origin)

            print ("+++++++++++++++++++++")
            # get list label of each file
            labels, sents_nolabel = text_utils.separate_label_sent(sentences_origin)
            arr_labels += labels
            # add to list sents of cluster
            sents_of_clus += sentences_stem

            document = text_utils.get_doc_from_sentences(sentences)

            bi_document = text_utils.convert_uni_to_bi([document])[0]
            print(sents_nolabel)
            arr_vectors += extract_feature.get_all_feature_from_doc(sents_nolabel, all_idf, bi_all_idf)

        # print(len(arr_vectors))
        # print(arr_vectors)




















        exit()
        #print(sents_of_clus)

        print(len( nmf_summarizer.getNMF(sents_of_clus)))
        # arr_scores_Lexrank = LexRank.getLexRank(sents_of_clus)
        # arr_scores_textrank = TextRank.getTextRank(sents_of_clus)
        # print(arr_scores_NMF)
        # print(arr_scores_Lexrank)
        # print(arr_scores_textrank)
        exit()

        # for i in range(len(sents_of_clus)):
        #     feature = arr_vectors[i] + ' 10:' + arr_scores_NMF[i] + ' 11:' + arr_scores_Lexrank[i]
        #     all_features.append(feature)
