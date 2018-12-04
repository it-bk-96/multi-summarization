# -*- coding: utf-8 -*-
# author Siddharth Sharma
# implementation of TextRank https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
# experiment: it would be interesting to use word2vec representation for words and cosine similarity to see how it performs

import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
if len(sys.argv) <= 2:
	print ("usage: testRankWord 3 document.txt")
	exit()
else:
	k = int(sys.argv[1])
	file = open(sys.argv[2],'r')
	text = file.read()


#text = "In Our Time is Ernest Hemingway's first collection of short stories, published in 1925 by Boni and Liveright, New York. Its title is derived from the English Book of Common Prayer, \"Give peace in our time, O Lord\". The collection's publication history was complex. It began with six prose vignettes commissioned by Ezra Pound for a 1923 edition of The Little Review. Hemingway added twelve more and in 1924 compiled the in our time edition (with a lower-case title), which was printed in Paris. To these were added fourteen short stories for the 1925 edition, including \"Indian Camp\" and \"Big Two-Hearted River\", two of his best-known Nick Adams stories. He composed \"On the Quai at Smyrna\" for the 1930 edition. The stories' themes – of alienation, loss, grief, separation – continue the work Hemingway began with the vignettes, which include descriptions of acts of war, bullfighting and current events. The collection is known for its spare language and oblique depiction of emotion, through a style known as Hemingway's \"theory of omission\" (Iceberg Theory). According to his biographer Michael Reynolds, among Hemingway's canon, \"none is more confusing ... for its several parts – biographical, literary, editorial, and bibliographical – contain so many contradictions that any analysis will be flawed.\" Hemingway's writing style attracted attention with literary critic Edmund Wilson saying it was \"of the first destinction\"; the 1925 edition of In Our Time is considered one of Hemingway's early masterpieces."
text = text.strip().replace('\r\n', ' ')

tagged = pos_tag(word_tokenize(text))

stop_words = set(stopwords.words("english"))
#print stop_words

#use noun and adj for keywords
filter_set = set(["NN","NNP","NNS","NNPS","JJ","JJR","JJS"])

tag_filtered = [(word.lower(),tag) for (word,tag) in tagged if (word.lower() not in stop_words and tag in filter_set and word.isalpha())]

words_filtered = [word for (word,tag) in tag_filtered]
print(words_filtered)

graph = {}
node_weight = {}

def add_edge(graph,node_weight,edge1,edge2,weight=1):
	if edge1 in graph:
		if edge2 not in graph[edge1]:
			graph[edge1].append(edge2)
	if edge1 not in graph:
		graph[edge1] = [edge2]

	if edge2 in graph:
		if edge1 not in graph[edge2]:
			graph[edge2].append(edge1)
	if edge2 not in graph:
		graph[edge2] = [edge1]
	
	node_weight[edge1] = 1
	node_weight[edge2] = 1

#creating graph
span = 3   # gioi han trong 2 word ke tiep => tao graph voi tu dang xet
i = 0
while i < len(words_filtered):
	j=i+1
	while j < i+span and j < len(words_filtered):
		add_edge(graph,node_weight,words_filtered[i],words_filtered[j]) 
		j+=1
	i+=1

#print graph
#print node_weight

#textrank begins here
def text_rank_word(graph,node_weight,d=.85,iterations=20):
	while iterations>0:
		for node in node_weight:
			temp=0.0
			for neighbour in graph[node]:
				temp+=node_weight[neighbour]/len(graph[neighbour])
			node_weight[node] = 1-d+(d*temp) 

		iterations-=1


text_rank_word(graph,node_weight,iterations=50)
#print node_weight

def my_fun(tup):
	return tup[1]

top_k_words = sorted(node_weight.items(),key=my_fun,reverse=True)[:k]

print(top_k_words)