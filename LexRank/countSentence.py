import os
import nltk
import re

def countSentences(file_path_and_name):
	f = open(file_path_and_name, 'r')
	text_0 = f.read()

	# code 2007
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

	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	lines = sent_tokenizer.tokenize(text_1.strip())
	return len(lines)

sum_all = 0

main_folder_path = os.getcwd() + "/Data_DUC_2004/Documents/"
for folder in os.listdir(main_folder_path):
	currentFolder = main_folder_path + folder + "/"
	sum = 0
	for file in os.listdir(currentFolder):
		count = countSentences(currentFolder + file)
		sum += count
	sum_all += sum/10
print(sum_all/50)
