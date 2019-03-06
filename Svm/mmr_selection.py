import text_utils
import math


# doc = {0: "Cao Mạnh Hải và Vũ Trường Giang ở Phú Thọ", 1: "Anh Duy Hiếu ở Thái Bình", 2: "doc3"}
# sent_orign = [{"doc": 0, "value": "Hải Phú Thọ", "score_svm": 0.92},
#               {"doc": 0, "value": "Giang Phú Thọ", "score_svm": 0.67},
#               {"doc": 1, "value": "Hiếu Thái Bình", "score_svm": 0.45}]
#
# idf = {"Hải": 1, "Phú": 2, "Thọ": 2, "Giang": 1, "Hiếu": 1, "Thái": 1, "Bình": 1}

def sentence_sim(sentence1, sentence2, idf, doc):
    numerator = 0
    denom1 = 0
    denom2 = 0

    try:
        for word in sentence2["value"].split(" "):

            tf_word_s2 = text_utils.get_word_freq(word, doc[sentence2["doc"]])
            if word in idf:
                numerator += (tf_word_s2 * text_utils.get_word_freq(word, doc[sentence1["doc"]])) * (idf[word] ** 2)
                denom2 += (text_utils.get_word_freq(word, doc[sentence2["doc"]]) * idf[word]) ** 2
            else:
                numerator += (tf_word_s2 * text_utils.get_word_freq(word, doc[sentence1["doc"]])) * (0.1 ** 2)
                denom2 += (text_utils.get_word_freq(word, doc[sentence2["doc"]]) * 0.1) ** 2

        for word in sentence1["value"].split(" "):
            if word in idf:
                idf_w = idf[word]
            else:
                idf_w = 0.1

            denom1 += (text_utils.get_word_freq(word, doc[sentence1["doc"]]) * idf_w) ** 2

        return numerator / (math.sqrt(denom2) * math.sqrt(denom1))
    except ZeroDivisionError:
        return float('-inf')


# Si,j = [{"doc": index, "value": sent}, ...]
def MMRScore(Si, summari, lambta, idf, doc):
    l_expr = lambta * Si["score_svm"]
    value = [float("-inf")]

    for sent in summari:
        # for each sent in summari, we compute similarity with new sent
        Sim2 = sentence_sim(Si, sent, idf, doc)
        value.append(Sim2)

    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr

    return MMR_SCORE


def make_summary(sentences, summary_length, lambta, IDF, doc):
    summary = [sentences[0]]
    del sentences[0]
    sum_len = len(sentences[0]["value"].split(' '))

    # keeping adding sentences until number of words exceeds summary length
    # print ('------------------------------')
    while (sum_len < summary_length - 5):

        index = -1
        max_score_mmr = -1
        for i in range(len(sentences)):
            sent = sentences[i]
            mmr_score = MMRScore(sent, summary, lambta, IDF, doc)
            if mmr_score > max_score_mmr:
                max_score_mmr = mmr_score
                index = i

        sent_selected = sentences[index]
        summary.append(sent_selected)

        del sentences[index]
        # print (sentences)
        if len(sentences) == 0:
            break
        sum_len += len(sent_selected['value'].split(' '))


    return summary

# if __name__ == "__main__":
#     a = make_summary(sent_orign, 20, 0.6, idf, doc)
#
#     print (a)
