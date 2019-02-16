# import os
# import nltk
#
# def countSentences(file_path_and_name):
# 	# Đọc file
# 	f = open(file_path_and_name, 'r')
# 	text_0 = f.read()
#
# 	# tách câu
# 	# lines = text_0.split("\n")
#
# 	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# 	lines = sent_tokenizer.tokenize(text_0.strip())
# 	return len(lines)
#
# # term = "summary_%s"
# # sum_all = 0
# #
# # main_folder_path = os.getcwd() + "/Data_DUC_2007/Human_Summaries/"
# # for i in range(1, 46):
# # 	id = "%02d" % (i)
# # 	print(term % id)
# # 	sum = 0
# # 	for X in ["A", "B", "C", "D"]:
# # 		count = countSentences(main_folder_path + (term % id) + "." + X + ".1.txt")
# # 		sum += count
# # 	print(sum/4)
# # 	sum_all += sum/4
# # print(sum_all/45)
#
# term = "cluster_%s.ref"
# sum_all = 0
#
# main_folder_path = os.getcwd() + "/Data_Chưa_tách_từ/Human_Summaries/"
# for i in list(range(1, 178)) + list(range(179, 201)):
# 	id = str(i)
# 	sum = 0
# 	for X in ["1", "2"]:
# 		count = countSentences(main_folder_path + (term % id) + X + ".txt")
# 		sum += count
# 	print(sum/2)
# 	sum_all += sum/2
# print(sum_all/199)


text = "anh đi về đêm khuya"
line = text.strip().lower()
from pyvi import ViTokenizer
# tách từ
stemmed_sentence = ViTokenizer.tokenize(line)
stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'"
										 and x != '!' and x != '''"''' and x != "''" and x != "'s",
							   stemmed_sentence))
print(stemmed_sentence)