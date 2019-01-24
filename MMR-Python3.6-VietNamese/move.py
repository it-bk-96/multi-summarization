import os
import shutil
import re

# src = '/home/gvt/Desktop/multi-summarization/MMR-Python3.6-VietNamese/Data_VN_200_cluster'
#
# for i in range(1, 201):
# 	os.makedirs(src + "/Documents" + "/cluster_" + str(i))
# 	os.makedirs(src + "/Human_Summaries" + "/cluster_" + str(i))

src_vn = '/home/gvt/Desktop/multi-summarization/MMR-Python3.6-VietNamese/Data_VN_200_cluster/Human_Summaries'
src = '/home/gvt/Desktop/multi-summarization/MMR-Python3.6-VietNamese/Data_200_Cluster_VN_Raw'

regex_string = r"\w+\.ref\d\.tok\.txt$"

for folder in os.listdir(src):
	src_folder = src + '/' + folder
	src_folder_vn = src_vn + '/' + folder

	for file in os.listdir(src_folder):
		if re.match(regex_string, file):
			src_file = src_folder + '/' + file
			shutil.copy(src_file, src_folder_vn)