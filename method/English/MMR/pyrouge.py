from __future__ import division
import collections
import os
import six
import nltk

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
    peer_counter = _ngram_counts(peer, n)
    # model do người viết nên có thể có 1 hoặc nhiều bản tóm tắt
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
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


root_directory = "/home/giangvu/Desktop/multi-summarization/"
if __name__ == "__main__":
    porter = nltk.PorterStemmer()

    system_path = root_directory + "Data/DUC_2007/TextRank_results/"
    human_path = root_directory + "Data/DUC_2007/Human_Summaries/"
    print("-" * 100)

    arr_rouge_precision = []
    arr_rouge_recall = []
    arr_rouge_f1 = []
    for file in os.listdir(system_path):
        file_name_system = system_path + file
        text_system = nltk.word_tokenize(open(file_name_system, 'r').read().strip())
        text_system = [porter.stem(word) for word in text_system]
        text_system = list(filter(lambda x: x != '.' and x != '`' and x != ','
                                            and x != '?' and x != "'" and x != '!' and x != '''"'''
                                            and x != "''" and x != "'s", text_system))

        file_name_human_1 = human_path + "summary_" + file[3:5] + ".A.1.txt"
        file_name_human_2 = human_path + "summary_" + file[3:5] + ".B.1.txt"
        file_name_human_3 = human_path + "summary_" + file[3:5] + ".C.1.txt"
        file_name_human_4 = human_path + "summary_" + file[3:5] + ".D.1.txt"
        arr_text_human = []
        for el in [file_name_human_1, file_name_human_2, file_name_human_3, file_name_human_4]:
            llll = nltk.word_tokenize(open(el, 'r').read().strip())

            # stemming words // đưa về từ gốc
            stemmedSent = [porter.stem(word) for word in llll]
            stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ','
                                                and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                and x != "''" and x != "'s", stemmedSent))
            arr_text_human.append(stemmedSent)

        arr_rouge_precision.append(rouge_1(text_system, arr_text_human, 1))
        arr_rouge_recall.append(rouge_1(text_system, arr_text_human, 0))
        arr_rouge_f1.append(rouge_1(text_system, arr_text_human, 0.5))

    mean_precision = sum(arr_rouge_precision) / len(arr_rouge_precision)
    mean_recall = sum(arr_rouge_recall) / len(arr_rouge_recall)
    mean_f1 = sum(arr_rouge_f1) / len(arr_rouge_f1)
    print("Rouge-1 với system và human: ")
    print("%.10s" % ("precision") + " | " + "%.10s" % ("recall") + " | " + "%.10s" % ("F1"))
    print("%.8f" % (mean_precision), "%.7f" % (mean_recall),
          "%.7f" % (mean_f1))