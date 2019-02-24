from rouge_vn import pyrouge_vn
from convert_to_extract import text_utils
import os

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"

class ConvertExtract(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def convert_extract(self, list_documents, list_human, path_save):
        sentences_origin_system, sentences_system, sentences_reference = \
            text_utils.get_all_sentences(list_documents, list_human)

        old_rouge = 0
        rouge = 0
        old_index = -1

        sentences = []
        sentences_origin = []

        while (0 == 0):
            i = 0
            for item in sentences_system:
                tmp = ""
                for word in sentences:
                    tmp += word

                tmp += item

                #Use rouge 2
                tmp_rouge_f1 = pyrouge_vn.rouge_1(tmp, sentences_reference, self.alpha)

                if tmp_rouge_f1 > rouge:
                    rouge = tmp_rouge_f1
                    old_index = i

                i += 1

            if rouge == old_rouge:
                break
            else:
                # print(rouge)
                old_rouge = rouge
                sentences.append(sentences_system[old_index])
                sentences_origin.append(sentences_origin_system[old_index])

                del sentences_system[old_index]
                del sentences_origin_system[old_index]

                old_index = -1

        for item in sentences_origin:
            with open(path_save, 'a') as file:
                file.write(item)

        tmp = ""
        for item in sentences_origin:
            tmp += item

        tmp1 = ""
        for item in sentences_reference:
            tmp1 += item

        with open("test_convert.txt", 'a') as txt_file:
            txt_file.write(str(rouge) + "\n")
            txt_file.write(tmp1)
            txt_file.write("\n\n-----------------------------------------\n\n")
            txt_file.write(tmp)
            txt_file.write("\n\n#############################################\n\n")

        print (rouge)
        return rouge

if __name__ == "__main__":
    convert_extract = ConvertExtract()
    #
    # list_documents = [DIR_PATH + "convert_to_extract/document1.txt",
    #                   DIR_PATH + "convert_to_extract/document2.txt",
    #                   DIR_PATH + "convert_to_extract/document3.txt"]
    # list_humans = [DIR_PATH + "convert_to_extract/human1.txt"]
    #
    # convert_extract.convert_extract(list_documents, list_humans, DIR_PATH + "convert_to_extract/test1.txt")
    #
    folder = "/home/hai/multi_document_summarization/github/data/BaoMoi"
    sum = 0.0
    count = 0
    for item in os.listdir(folder + "/bao_moi_process/documents/0"):
        # print (item)
        # with open(folder + "/bao_moi_process/documents/0/" + item) as doc_file:
        #     document = doc_file.read()
        #
        # with open(folder + "/bao_moi_process/summaries/abstract/0/" + item) as abstract_file:
        #     summary = abstract_file.read()
        #
        # print (11111111111111111111)
        sum += convert_extract.convert_extract([folder + "/bao_moi_process/documents/0/" + item],
                                               [folder + "/bao_moi_process/summaries/abstract/0/" + item],
                                               folder + "/bao_moi_process/summaries/extract/0/" + item)
        count += 1

    print ("-----------------------------")
    print (sum/count)
