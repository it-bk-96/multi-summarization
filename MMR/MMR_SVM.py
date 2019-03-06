from extract_feature_svm import text_utils
import math

doc = {0: "Cao Mạnh Hải và Vũ Trường Giang ở Phú Thọ", 1: "Anh Duy Hiếu ở Thái Bình", 2: "doc3"}
sent_orign = [{"doc": 0, "value": "Hải Phú Thọ", "score_svm": 0.92},
              {"doc": 0, "value": "Giang Phú Thọ", "score_svm": 0.67},
              {"doc": 1, "value": "Hiếu Thái Bình", "score_svm": 0.45}]

idf = {"Hải": 1, "Phú": 2, "Thọ": 2, "Giang": 1, "Hiếu": 1, "Thái": 1, "Bình": 1}

def sentence_sim(sentence1, sentence2, idf, doc):
    numerator = 0
    denominator = 0

    for word in sentence2["value"].split(" "):
        numerator += text_utils.get_word_freq(word, doc[sentence1["doc"]]) * text_utils.get_word_freq(word, doc[sentence2["doc"]]) * idf[word]**2

    for word in sentence1["value"].split(" "):
        denominator += (text_utils.get_word_freq(word, doc[sentence1["doc"]]) * idf[word]) ** 2

    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")

#Si,j = [{"doc": index, "value": sent}, ...]
def MMRScore(Si, Sj, lambta, idf, doc):
    l_expr = lambta * Si["score_svm"]
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentence_sim(Si, sent, idf, doc)
        value.append(Sim2)

    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr

    return MMR_SCORE

def make_summary(sentences, summary_length, lambta, IDF, doc):
    summary = [sentences[0]]
    del sentences[0]
    sum_len = len(sentences[0]["value"])

    # keeping adding sentences until number of words exceeds summary length
    # print ('------------------------------')
    while (sum_len < summary_length):

        selected_sent = 0
        old_score_mmr = -1
        old_index = -1

        i = 0
        for sent in sentences:
            mmr_score = MMRScore(sent, summary, lambta, IDF, doc)

            # print (mmr_score)
            if mmr_score > old_score_mmr:
                old_score_mmr = mmr_score
                selected_sent = sent
                old_index = i

            i += 1
            # print ("----------------")
        # print (selected_sent)
        # print MMRval[maxxer], maxxer.originalWords
        summary.append(selected_sent)

        del sentences[old_index]
        print (sentences)
        if len(sentences) == 0:
            break
        # print(sentences)
        sum_len += len(selected_sent["value"])

    # print ('---------------------------------')
    return summary

if __name__ == "__main__":
    a = make_summary(sent_orign, 20, 0.6, idf, doc)

    print (a)