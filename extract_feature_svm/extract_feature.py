from extract_feature_svm import text_utils
import os

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

    #FreqWord_Uni, FreqWord_Bi
    def extract_feature7_8(self, sentence, document):
        freq_words = text_utils.get_freq_word_uni(document)

        feature = 0.0
        for item in sentence.split(' '):
            if item not in freq_words:
                continue
            else:

                feature += freq_words[item]

        return feature

    #Centroid_Uni, Centroid_Bi
    def extract_feature5_6(self, sentence, document, all_idf):

        centroid_uni = text_utils.get_centroid_uni(document, all_idf)

        feature = 0.0
        for item in sentence.split(' '):
            if item not in centroid_uni:
                continue
            else:
                feature += centroid_uni[item]

        return feature

    #FirstRel_Doc
    def extract_feature9(self, sentence, all_idf, sentences):
        return text_utils.cos_similarity(sentence, all_idf, sentences)

if __name__ == "__main__":
    stop_words = text_utils.read_stopwords(DIR_PATH + "extract_feature_svm/stop_words.txt")
    documents = [DIR_PATH + "extract_feature_svm/document1.txt", DIR_PATH + "extract_feature_svm/document2.txt"]
    documents = text_utils.read_all_documents(documents, stop_words)
    extract_feature = FeatureSvm(DIR_PATH + "extract_feature_svm/document1.txt",
                                 DIR_PATH + "extract_feature_svm/stop_words.txt")

    sentences = extract_feature.get_sentences()
    document = text_utils.get_doc_from_sentences(sentences)

    bi_document = text_utils.convert_uni_to_bi([document])[0]
    bi_documents = text_utils.convert_uni_to_bi(documents)

    all_idf = text_utils.get_all_idf(documents)

    # print (all_idf)
    bi_all_idf = text_utils.get_all_idf(bi_documents)

    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    feature6 = []
    feature7 = []
    feature8 = []
    feature9 = []
    for item in sentences:
        bi_sentence = text_utils.convert_uni_to_bi([item])[0]

        feature1.append(extract_feature.extract_feature1(item, sentences))
        feature2.append(extract_feature.extract_feature2(item))
        feature3.append(extract_feature.extract_feature3(item))
        feature4.append(extract_feature.extract_feature4(item))
        feature5.append(extract_feature.extract_feature5_6(item, document, all_idf))
        feature6.append(extract_feature.extract_feature5_6(bi_sentence, bi_document, bi_all_idf))
        feature7.append(extract_feature.extract_feature7_8(item, document))
        feature8.append(extract_feature.extract_feature7_8(bi_sentence, bi_document))
        feature9.append(extract_feature.extract_feature9(item, all_idf, sentences))

    print (feature1)
    print (feature2)
    print (feature3)
    print (feature4)
    print (feature5)
    print (feature6)
    print (feature7)
    print (feature8)
    print (feature9)
