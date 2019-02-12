import os
import shutil

src = '/home/gvt/Desktop/multi-summarization/MMR/Data_DUC_2007/Human_Summaries'
for folder in os.listdir(src):
	src_folder = src + '/' + folder
	for file in os.listdir(src_folder):
		src_file = src_folder + '/' + file
		shutil.copy(src_file, src)