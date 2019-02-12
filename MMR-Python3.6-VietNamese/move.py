import os
import shutil
import re

# src = '/home/gvt/Desktop/multi-summarization/MMR-Python3.6-VietNamese/Data_Chưa_tách_từ'
#
# for i in range(1, 201):
# 	os.makedirs(src + "/Documents" + "/cluster_" + str(i))


src_vn = '/home/giangvu/Desktop/multi-summarization/MMR-Python3.6-VietNamese/Data_Chưa_tách_từ/Human_Summaries'
src = '/home/giangvu/Desktop/multi-summarization/MMR-Python3.6-VietNamese/Data_200_Cluster_VN_Raw'

regex_string = r"\w+\.ref\d\.txt$"
# regex_string = r"^\d+\.body\.txt$"

for folder in os.listdir(src):
	src_folder = src + '/' + folder
	src_folder_vn = src_vn + '/' + folder

	for file in os.listdir(src_folder):
		if re.match(regex_string, file):
			src_file = src_folder + '/' + file
			shutil.copy(src_file, src_vn)