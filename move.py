import re
import nltk
# import os


def processFile(file_name):
    # read file from provided folder path
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    f = open(file_name, 'r')
    text_0 = f.read()

    # extract content in TEXT tag and remove tags
    # code 2004
    # text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
    # text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
    # text_1 = re.sub("\n</TEXT>", "", text_1)
    #
    # # replace all types of quotations by normal quotes
    # text_1 = re.sub("\n", " ", text_1)
    #
    # text_1 = re.sub("\"", "\"", text_1)
    # text_1 = re.sub("''", "\"", text_1)
    # text_1 = re.sub("``", "\"", text_1)
    #
    # text_1 = re.sub(" +", " ", text_1)

    # code 2007
    text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
    text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
    text_1 = re.sub("\n</TEXT>", "", text_1)

    # replace all types of quotations by normal quotes
    text_1 = re.sub("<P>", "", text_1)
    text_1 = re.sub("</P>", "", text_1)
    text_1 = re.sub("\n", " ", text_1)

    text_1 = re.sub("\"", "\"", text_1)
    text_1 = re.sub("''", "\"", text_1)
    text_1 = re.sub("``", "\"", text_1)

    text_1 = re.sub(" +", " ", text_1)


    # segment data into a list of sentences //  tách câu
    lines = sentence_token.tokenize(text_1.strip())

    document = " ".join(lines)

    return document

#
# if __name__ == '__main__':
#
#     # set the main Document folder path where the subfolders are present
#     # main_folder_path = os.getcwd() + "/DUC_2004/Documents"
#     main_folder_path = os.getcwd() + "/Data/DUC_2007/Documents"
#
#     # read in all the subfolder names present in the main folder
#     for folder in os.listdir(main_folder_path):
#
#         print("Running MMR Summarizer for files in folder: ", folder)
#         # for each folder run the MMR summarizer and generate the final summary
#         curr_folder = main_folder_path + "/" + folder
#         path = os.getcwd() + "/Data/DUC_2007/Documents_New/" + folder
#         os.mkdir(path)
#
#         # find all files in the sub folder selected
#         files = os.listdir(curr_folder)
#         for file in files:
#             document = processFile(curr_folder + "/" + file)
#             results_folder = path + "/"
#             # print(os.path.join(results_folder, (str(file))))
#             with open(os.path.join(results_folder, (str(file))), "w") as fileOut:
#                 fileOut.write(document)

print([1,2, 3 ,4 ][:2])
print([1,2, 3 ,4 ][2:])