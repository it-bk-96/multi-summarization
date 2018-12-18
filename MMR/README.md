Create virtualenv on Python 2.7
Then install :
numpy : pip install numpy
bs4 : pip install beautifulsoup4
nltk : pip install -U nltk

To run pyrouge155 :
Remove :
rm ROUGE-1.5.5/data/WordNet-2.0.exc.db
Then create new : 
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db