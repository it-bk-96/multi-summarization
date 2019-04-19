import os
import re
import nltk

def processFile(file_name):
    # read file from provided folder path
    f = open(file_name, 'r')
    text_0 = f.read()

    text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
    text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
    text_1 = re.sub("\n</TEXT>", "", text_1)

    text_1 = re.sub("<P>", "", text_1)
    text_1 = re.sub("</P>", "", text_1)
    text_1 = re.sub("\n", " ", text_1)
    text_1 = re.sub("\"", "\"", text_1)
    text_1 = re.sub("''", "\"", text_1)
    text_1 = re.sub("``", "\"", text_1)
    text_1 = re.sub(" +", " ", text_1)
    text_1 = re.sub(" _ ", "", text_1)

    text_1 = re.sub(r"\(AP\) _", " ", text_1)
    text_1 = re.sub("&\w+;", " ", text_1)

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = sent_tokenizer.tokenize(text_1.strip())
    # setting the stemmer

    # preprocess line[0]

    index = lines[0].find("--")
    if index != -1:
        lines[0] = lines[0][index + 2:]
    index = lines[0].find(" _ ")
    if index != -1:
        lines[0] = lines[0][index + 3:]

    document = " ".join(lines)

    return document


if __name__ == '__main__':

    main_folder_path = os.getcwd() + "/Data/DUC_2007/Documents"

    for folder in os.listdir(main_folder_path):

        print(folder)
        curr_folder = main_folder_path + "/" + folder
        path = os.getcwd() + "/Data/DUC_2007/Documents_process/" + folder
        os.mkdir(path)

        # find all files in the sub folder selected
        files = os.listdir(curr_folder)
        for file in files:
            document = processFile(curr_folder + "/" + file)
            results_folder = path + "/"
            # print(os.path.join(results_folder, (str(file))))
            with open(os.path.join(results_folder, (str(file))), "w") as fileOut:
                fileOut.write(document)

# print([1,2, 3 ,4 ][:2])
# print([1,2, 3 ,4 ][2:])

# import json
# #
# with open('/home/giangvu/Desktop/multi-summarization/method/VietNamese/SVM/Data/converted/cluster_13/12235782.body.txt') as json_file:
#     datas = json.load(json_file)
#     for data in datas:
#         print(data['label'])
#         print(data['tokenized'])
#         print(data['origin'])