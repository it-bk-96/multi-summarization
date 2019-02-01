python 3.6:
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

Rouge-1 với system và human chưa tách từ textRank - 200 cluster: 
precision | recall | F1
0.34707538 0.4180899 0.3716248
----------------------------------------
Execution time: 0.39081692695617676