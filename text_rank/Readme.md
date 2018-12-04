These scripts were an attempt made by me to understand text summarization. I implemented two papers which used unsupervised method to extract the most important keywords/sentences from text. The basic idea of these papers was to create a graph associated with text where the vertices represent the entity to be ranked and the edges indicate some relationship(which can be syntactic or semantic) between vertices, and then PageRank algorithm is used to rank these vertices.

The two papers that I explored were:
    LexRank: www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html
    TextRank: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

I further plan on exploring papers related to Multi-document summarization; specifically:
    R. McDonald. A Study of Global Inference Algorithms in Multi-Document Summarization ECIR 2007. (formulates summarization task as global optimization problem using integer linear programming)
    W. Yih et al. Multi-Document Summarization by Maximizing Informative Content-Words. IJCAI 2007. (introduces stack decoding to this field)

Scripts:
    testRankWord.py : implements textRank algorithm for keyword extraction.
    testRankSent.py : implements textRank algorithm for sentence summarizaton.
    lexRank.py : implements lexrank algorithm for sentence summarization.

usage:
    script_name number_of_top_entities document_containing_text
    example - ./lexrank.py 3 data.txt

Dependency:
	Nltk, numpy