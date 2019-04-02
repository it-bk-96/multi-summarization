from __future__ import division
import os
import re
import six
import nltk
import collections

porter = nltk.PorterStemmer()
root_abtract = os.path.dirname(os.getcwd())


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


def getRougeEnglish(number, system, humans):
    text_system = nltk.word_tokenize(system.strip())
    text_system = [porter.stem(word) for word in text_system]
    text_system = list(filter(lambda x: x != '.' and x != '`' and x != ','
                                        and x != '?' and x != "'" and x != '!' and x != '''"'''
                                        and x != "''" and x != "'s", text_system))
    arr_text_human = []
    for human in humans:
        llll = nltk.word_tokenize(human.strip())
        stemmedSent = [porter.stem(word) for word in llll]
        stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ','
                                            and x != '?' and x != "'" and x != '!' and x != '''"'''
                                            and x != "''" and x != "'s", stemmedSent))
        arr_text_human.append(stemmedSent)

    precision = 0
    recall = 0
    f1 = 0
    if number == 1:
        precision = (rouge_1(text_system, arr_text_human, 1))
        recall = (rouge_1(text_system, arr_text_human, 0))
        f1 = (rouge_1(text_system, arr_text_human, 0.5))
    if number == 2:
        precision = (rouge_2(text_system, arr_text_human, 1))
        recall = (rouge_2(text_system, arr_text_human, 0))
        f1 = (rouge_2(text_system, arr_text_human, 0.5))
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def getRougeAverageEnglish(method, type, id):
    root_directory = os.path.dirname(os.getcwd())
    porter = nltk.PorterStemmer()
    id = str(id)
    method = str(method)
    type = str(type)

    list_method = {
        'MMR': "MMR_results/"
        ,
        'LexRank': "LexRank_results/"
        ,
        'TextRank': "TextRank_results/"
        ,
        'NMF': "NMF_results/"
        ,
    }
    try:
        system_path = root_abtract + "/Data/DUC_2007/" + list_method[method]
        human_path = root_abtract + "/Data/DUC_2007/Human_Summaries/"
    except:
        return {
            'rouge_1': {
                'precision': 0,
                'recall': 0,
                'f1': 0,
            },
            'rouge_2': {
                'precision': 0,
                'recall': 0,
                'f1': 0,
            },
        }

    arr_rouge_precision_1 = []
    arr_rouge_recall_1 = []
    arr_rouge_f1_1 = []
    arr_rouge_precision_2 = []
    arr_rouge_recall_2 = []
    arr_rouge_f1_2 = []

    if type == 'all':
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

            arr_rouge_precision_1.append(rouge_1(text_system, arr_text_human, 1))
            arr_rouge_recall_1.append(rouge_1(text_system, arr_text_human, 0))
            arr_rouge_f1_1.append(rouge_1(text_system, arr_text_human, 0.5))

            arr_rouge_precision_2.append(rouge_1(text_system, arr_text_human, 1))
            arr_rouge_recall_2.append(rouge_1(text_system, arr_text_human, 0))
            arr_rouge_f1_2.append(rouge_1(text_system, arr_text_human, 0.5))

    if type == 'topic':
        id_format = "%02d" % int(id)
        file_name_system = ""
        for file in os.listdir(system_path):
            if re.search(r"^D07" + "%02d" % int(id), file):
                file_name_system = system_path + "/" + file
                break
        text_system = nltk.word_tokenize(open(file_name_system, 'r').read().strip())
        text_system = [porter.stem(word) for word in text_system]
        text_system = list(filter(lambda x: x != '.' and x != '`' and x != ','
                                            and x != '?' and x != "'" and x != '!' and x != '''"'''
                                            and x != "''" and x != "'s", text_system))

        file_name_human_1 = human_path + "summary_" + id_format + ".A.1.txt"
        file_name_human_2 = human_path + "summary_" + id_format + ".B.1.txt"
        file_name_human_3 = human_path + "summary_" + id_format + ".C.1.txt"
        file_name_human_4 = human_path + "summary_" + id_format + ".D.1.txt"
        arr_text_human = []
        for el in [file_name_human_1, file_name_human_2, file_name_human_3, file_name_human_4]:
            llll = nltk.word_tokenize(open(el, 'r').read().strip())

            # stemming words // đưa về từ gốc
            stemmedSent = [porter.stem(word) for word in llll]
            stemmedSent = list(filter(lambda x: x != '.' and x != '`' and x != ','
                                                and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                and x != "''" and x != "'s", stemmedSent))
            arr_text_human.append(stemmedSent)

        arr_rouge_precision_1.append(rouge_1(text_system, arr_text_human, 1))
        arr_rouge_recall_1.append(rouge_1(text_system, arr_text_human, 0))
        arr_rouge_f1_1.append(rouge_1(text_system, arr_text_human, 0.5))

        arr_rouge_precision_2.append(rouge_1(text_system, arr_text_human, 1))
        arr_rouge_recall_2.append(rouge_1(text_system, arr_text_human, 0))
        arr_rouge_f1_2.append(rouge_1(text_system, arr_text_human, 0.5))


    return {
        'rouge_1': {
            'precision': sum(arr_rouge_precision_1) / len(arr_rouge_precision_1),
            'recall': sum(arr_rouge_recall_1) / len(arr_rouge_recall_1),
            'f1': sum(arr_rouge_f1_1) / len(arr_rouge_f1_1),
        },
        'rouge_2': {
            'precision': sum(arr_rouge_precision_2) / len(arr_rouge_precision_2),
            'recall': sum(arr_rouge_recall_2) / len(arr_rouge_recall_2),
            'f1': sum(arr_rouge_f1_2) / len(arr_rouge_f1_2),
        },
    }
