# -*- coding: utf-8 -*-
# author Siddharth Sharma
# implementation of TextRank https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np

if len(sys.argv) <= 2:
	print("usage: testRankSent 3 document.txt")
	exit()
else:
	top_k = int(sys.argv[1])
	file = open(sys.argv[2],'r')
	text = file.read()

#text = "In Our Time is Ernest Hemingway's first collection of short stories, published in 1925 by Boni and Liveright, New York. Its title is derived from the English Book of Common Prayer, \"Give peace in our time, O Lord\". The collection's publication history was complex. It began with six prose vignettes commissioned by Ezra Pound for a 1923 edition of The Little Review. Hemingway added twelve more and in 1924 compiled the in our time edition (with a lower-case title), which was printed in Paris. To these were added fourteen short stories for the 1925 edition, including \"Indian Camp\" and \"Big Two-Hearted River\", two of his best-known Nick Adams stories. He composed \"On the Quai at Smyrna\" for the 1930 edition. The stories' themes – of alienation, loss, grief, separation – continue the work Hemingway began with the vignettes, which include descriptions of acts of war, bullfighting and current events. The collection is known for its spare language and oblique depiction of emotion, through a style known as Hemingway's \"theory of omission\" (Iceberg Theory). According to his biographer Michael Reynolds, among Hemingway's canon, \"none is more confusing ... for its several parts – biographical, literary, editorial, and bibliographical – contain so many contradictions that any analysis will be flawed.\" Hemingway's writing style attracted attention with literary critic Edmund Wilson saying it was \"of the first destinction\"; the 1925 edition of In Our Time is considered one of Hemingway's early masterpieces."
text = text.strip().replace('\r\n', ' ')

sentences = sent_tokenize(text)

stop_words = set(stopwords.words('english'))

def token_lower(sentence):
	return [word.lower() for word in word_tokenize(sentence) if (word not in stop_words and word.isalpha())]

tok_fil_sent = map(token_lower,sentences)
vkl = list(tok_fil_sent)

num_nodes = len(vkl)

graph = np.zeros((num_nodes,num_nodes)) #I doubt number of sentences will be large so I decided to go with this Adjacency matrice representation
for i in range(num_nodes):
	for j in range(i+1,num_nodes):   # tinh toan overlap of 2 sentences
		graph[i,j] = float(len(set(vkl[i])&set(vkl[j])))/(len(vkl[i])+len(vkl[j]))
		graph[j,i] = graph[i,j]
print(graph)

node_weights = np.ones(num_nodes)
#print node_weights

def text_rank_sent(graph,node_weights,d=.85,iter=20):
	weight_sum = np.sum(graph,axis=0)
	while iter >0:
		for i in range(len(node_weights)):
			temp = 0.0
			for j in range(len(node_weights)):
				temp += graph[i,j]*node_weights[j]/weight_sum[j]
			node_weights[i] = 1-d+(d*temp)
		iter-=1

text_rank_sent(graph,node_weights)

#print node_weights
#top_k = 3
top_index = [i for i,j in sorted(enumerate(node_weights), key=lambda x: x[1],reverse=True)[:top_k]]

top_sentences = [sentences[i] for i in top_index]
print(top_sentences)