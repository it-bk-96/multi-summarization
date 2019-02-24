import os

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"

folder = DIR_PATH + "data/BaoMoi"

# for item in os.listdir(folder):
#     with open(folder + "/" + item) as txt_file:
#         txt = txt_file.read()
#     # print (txt)
#     tmp = txt.split("#")
#     i = 0
#     for doc in tmp:
#         print (doc)
#
#         if i == 10:
#             break
#         i += 1
#
#     break

# with open(folder + "/BaoMoi/0001-baomoi-articles.txt") as txt_file:
#     txt = txt_file.read()
#
#     # i = 0
#     # for item in txt.split("#\n"):
#     #     if i == 0:
#     #         i += 1
#     #         continue
#     #
#     #     print (item)
#     #
#     #     i += 1
#
# b1 = txt.split("#\n")[1]
# test = b1.split("\n")[:-2]
#
# if len(test) <= 2:
#     print ("lll")
# else:
#     summary = test[1]
#     document = ""
#     for item in range(2, len(test)):
#         document += test[item] + "\n"
#
#     document = document[:-1]
#
#     print (summary)
#     with open(folder + "/bao_moi_process/summaries/abstract/" + "1_" + "0001-baomoi-articles.txt", "w") as summary_file:
#         summary_file.write(summary)
#
#     with open(folder + "/bao_moi_process/documents/" + "1_" + "0001-baomoi-articles.txt", "w") as document_file:
#         document_file.write(document)

def extract_summary_and_document(text):
    lines = text.split("\n")[:-1]

    if len(lines) <= 2:
        return "", ""

    summary = lines[1]

    document = ""
    for index_line in range(2, len(lines)):
        document += lines[index_line] + "\n"

    if len(summary) >= len(document[:-1]):
        return "", ""

    return summary, document[:-1]

j = 0
for name_file in os.listdir(folder + "/BaoMoi"):
    with open(folder + "/BaoMoi/" + name_file) as txt_file:
        text = txt_file.read()

    origin_documents = text.split("#\n")[1:]
    # print (len(origin_documents))
    i = 1
    # print (origin_documents[len(origin_documents)-1])
    for doc in origin_documents:

        summary, document = extract_summary_and_document(doc)

        if summary == "" or document == "":
            continue

        if not os.path.exists(folder + "/bao_moi_process/summaries/abstract/" + str(j)):
            os.mkdir(folder + "/bao_moi_process/summaries/abstract/" + str(j))
        if not os.path.exists(folder + "/bao_moi_process/documents/" + str(j)):
            os.mkdir(folder + "/bao_moi_process/documents/" + str(j))

        with open(folder + "/bao_moi_process/summaries/abstract/" + str(j) + "/" + str(i) + "_" + name_file, "w") as summary_file:
            summary_file.write(summary)

        with open(folder + "/bao_moi_process/documents/" + str(j) + "/" + str(i) + "_" + name_file, "w") as document_file:
            document_file.write(document)

        i += 1

    j += 1
