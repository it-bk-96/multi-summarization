from __future__ import division
import collections
import os
import six
import time
import nltk

# from LexRank.lexrank_summarizer import sentence


def _ngrams(words, n):
	queue = collections.deque(maxlen=n)
	for w in words:
		queue.append(w)
		if len(queue) == n:
			yield tuple(queue)


def _ngram_counts(words, n):
	return collections.Counter(_ngrams(words, n))


def _ngram_count(words, n):
	return max(len(words) - n + 1, 0)


def _counter_overlap(counter1, counter2):
	result = 0
	for k, v in six.iteritems(counter1):
		result += min(v, counter2[k])
	return result


def _safe_divide(numerator, denominator):
	if denominator > 0:
		return numerator / denominator
	else:
		return 0


def _safe_f1(matches, recall_total, precision_total, alpha):
	recall_score = _safe_divide(matches, recall_total)
	precision_score = _safe_divide(matches, precision_total)
	denom = (1.0 - alpha) * precision_score + alpha * recall_score
	if denom > 0.0:
		return (precision_score * recall_score) / denom
	else:
		return 0.0


def rouge_n(peer, models, n, alpha):
	"""
	Compute the ROUGE-N score of a peer with respect to one or more models, for
	a given value of `n`.
	"""
	matches = 0
	recall_total = 0
	peer_counter = _ngram_counts(peer.split(), n)
	# model do người viết nên có thể có 1 hoặc nhiều bản tóm tắt
	for model in models:
		model_counter = _ngram_counts(model.split(), n)
		matches += _counter_overlap(peer_counter, model_counter)
		recall_total += _ngram_count(model.split(), n)
	precision_total = len(models) * _ngram_count(peer.split(), n)
	return _safe_f1(matches, recall_total, precision_total, alpha)


def rouge_1(peer, models, alpha):
	"""
	Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
	models.
	"""
	return rouge_n(peer, models, 1, alpha)


def rouge_2(peer, models, alpha):
	"""
	Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
	models.
	"""
	return rouge_n(peer, models, 2, alpha)


def rouge_3(peer, models, alpha):
	"""
	Compute the ROUGE-3 (trigram) score of a peer with respect to one or more
	models.
	"""
	return rouge_n(peer, models, 3, alpha)


def lcs(a, b):
	"""
	Compute the length of the longest common subsequence between two sequences.
	Time complexity: O(len(a) * len(b))
	Space complexity: O(min(len(a), len(b)))
	"""
	# This is an adaptation of the standard LCS dynamic programming algorithm
	# tweaked for lower memory consumption.
	# Sequence a is laid out along the rows, b along the columns.
	# Minimize number of columns to minimize required memory
	if len(a) < len(b):
		a, b = b, a
	# Sequence b now has the minimum length
	# Quit early if one sequence is empty
	if len(b) == 0:
		return 0
	# Use a single buffer to store the counts for the current row, and
	# overwrite it on each pass
	row = [0] * len(b)
	for ai in a:
		left = 0
		diag = 0
		for j, bj in enumerate(b):
			up = row[j]
			if ai == bj:
				value = diag + 1
			else:
				value = max(left, up)
			row[j] = value
			left = value
			diag = up
	# Return the last cell of the last row
	return left


def rouge_l(peer, models, alpha):
	"""
	Compute the ROUGE-L score of a peer with respect to one or more models.
	"""
	matches = 0
	recall_total = 0
	for model in models:
		matches += lcs(model, peer)
		recall_total += len(model)
	precision_total = len(models) * len(peer)
	return _safe_f1(matches, recall_total, precision_total, alpha)

#haicm
from pyvi import ViTokenizer
import nltk

def split_sentences(file_name):
	with open(file_name, 'r') as file:
		text_system = file.read()

	sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
	tmp = sentence_token.tokenize(text_system)

	sentences = []
	for item in tmp:
		if "…" in item:
			b = item.split("…")
			for i in b:
				sentences.append(i)
		else:
			sentences.append(item)

	return sentences

def get_all_sentences(file_system, file_reference):
	sentences_origin_system = []
	for item in file_system:
		sentences_origin_system += split_sentences(item)

	sentences_system = []
	for item in sentences_origin_system:
		sentences_system.append(ViTokenizer.tokenize(item))

	sentences_origin_reference = []
	for item in file_reference:
		with open(item, 'r') as file:
			sentences_origin_reference.append(file.read())

	# for item in sentences_origin_reference:
	# 	print (item)

	sentences_reference = []
	for item in sentences_origin_reference:
		sentences_reference.append(ViTokenizer.tokenize(item))

	return sentences_origin_system, sentences_system, sentences_reference

if __name__ == "__main__":
	sentences_origin_system, sentences_system, sentences_reference = \
		get_all_sentences(["document1.txt", "document2.txt", "document3.txt"], ["human1.txt", "human2.txt"])

	# print (sentences_reference)

	old_rouge = 0
	rouge = 0
	old_index = -1

	sentences = []
	sentences_origin = []
	while (0 == 0):
		i = 0
		for item in sentences_system:
			tmp = ""
			for word in sentences:
				tmp += word

			tmp += item

			tmp_rouge_f1 = rouge_2(tmp, sentences_reference, 0.5)

			if tmp_rouge_f1 > rouge:
				rouge = tmp_rouge_f1
				old_index = i

			i += 1

		if rouge == old_rouge:
			break
		else:
			print (rouge)
			old_rouge = rouge
			sentences.append(sentences_system[old_index])
			sentences_origin.append(sentences_origin_system[old_index])
			old_index = -1

	print (sentences_origin)

	for item in sentences_origin:
		with open("result.txt", 'a') as file:
			file.write(item)
    # print (rouge_1("Con mèo đã được tìm thấy ở dưới cái giường", ["Con mèo đã ở dưới cái giường"], 0))
    # print (rouge_1("Con mèo đã được tìm thấy ở dưới cái giường", ["Con mèo đã ở dưới cái giường"], 1))
    # print (rouge_1("Con mèo đã được tìm thấy ở dưới cái giường", ["Con mèo đã ở dưới cái giường"], 0.5))

