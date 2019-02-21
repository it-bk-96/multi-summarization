from pyvi import ViTokenizer
from collections import Counter
from nltk.util import ngrams


a = list(ngrams(ViTokenizer.tokenize("Tôi yêu 30/10 cô ấy trái tim cô ấy").split(), 2))

print('ngày 23/10 lực_lượng đối_lập '.split())
