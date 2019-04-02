import flask
import re
import nltk

porter = nltk.PorterStemmer()

import os
from flask_cors import CORS
from flask import request, jsonify
from apiSummarization.Method.TextRank import TextRank
from apiSummarization.Method.MMR import mmr_summarizer
from apiSummarization.Method.LexRank import LexRank
from pyvi import ViTokenizer
# from NonNegativeMatrixFactorization import nmf_summarizer
from apiSummarization.rouge import getRouge, getRougeAverage
from apiSummarization.rougeEnglish import getRougeEnglish, getRougeAverageEnglish

from apiSummarization.score_feature.LexRank import getLexRank
from apiSummarization.score_feature.nmf_summarizer import getNMF
from apiSummarization.score_feature.TextRank import getTextRank
from apiSummarization.score_feature.mmr_summarizer import getMMR
from apiSummarization.score_feature.KMean import getKmeanPMMR

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True
sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Good morning</h1>
                <p>Get ready</p>'''


@app.route('/api/v1/resources/topic_method', methods=['GET'])
def api_get_topic_method():
    if 'topic_id' in request.args:
        topic_id = int(request.args['topic_id'])
        method = str(request.args['method'])
        language = str(request.args['language'])
        if topic_id == 178:
            return "Error: No topic_id field provided. Please specify an topic_id."
    else:
        return "Error: No topic_id field provided. Please specify an topic_id."

    array_document = {}
    result = {}
    documents = []
    i = 1
    text_1 = ""
    text_2 = ""
    if language == "vn":
        document_path = os.path.dirname(os.getcwd()) + "/Data/VietNamese/Documents/" + "cluster_" + str(topic_id) + "/"
        for file in os.listdir(document_path):
            file_name = document_path + file
            text = open(file_name, "rt", encoding="utf-8").read()
            documents.append(text)
            lines = sentence_token.tokenize(text)
            array_document[i] = [[0, line] for line in lines]
            i += 1
        file_human_1 = os.path.dirname(os.getcwd()) + "/Data/VietNamese/Human_Summaries/" + "cluster_" + str(topic_id) + ".ref1.txt"
        file_human_2 = os.path.dirname(os.getcwd()) + "/Data/VietNamese/Human_Summaries/" + "cluster_" + str(topic_id) + ".ref2.txt"
        text_1 = open(file_human_1, "rt", encoding="utf-8").read()
        text_2 = open(file_human_2, "rt", encoding="utf-8").read()

        result['human'] = {
            1: text_1,
            2: text_2
        }
    if language == "en":
        document_path = os.path.dirname(os.getcwd()) + "/Data/DUC_2007/Documents/"
        for folder in os.listdir(document_path):
            if re.search(r"^D07" + "%02d" % topic_id, folder):
                document_folder = document_path + folder + "/"
                for file in os.listdir(document_folder):
                    file_name = document_folder + file
                    lines = processFile__English(file_name)
                    documents.append(" ".join(lines))
                    array_document[i] = [[0, line] for line in lines]
                    i += 1
                break
        file_human_1 = os.path.dirname(
            os.getcwd()) + "/Data/DUC_2007/Human_Summaries/" + "summary_" + "%02d" % topic_id + ".A.1.txt"
        file_human_2 = os.path.dirname(
            os.getcwd()) + "/Data/DUC_2007/Human_Summaries/" + "summary_" + "%02d" % topic_id + ".A.1.txt"
        file_human_3 = os.path.dirname(
            os.getcwd()) + "/Data/DUC_2007/Human_Summaries/" + "summary_" + "%02d" % topic_id + ".A.1.txt"
        file_human_4 = os.path.dirname(
            os.getcwd()) + "/Data/DUC_2007/Human_Summaries/" + "summary_" + "%02d" % topic_id + ".A.1.txt"
        text_1 = open(file_human_1, "rt", encoding="utf-8").read()
        text_2 = open(file_human_2, "rt", encoding="utf-8").read()
        text_3 = open(file_human_3, "rt", encoding="utf-8").read()
        text_4 = open(file_human_4, "rt", encoding="utf-8").read()

        result['human'] = {
            1: text_1,
            2: text_2,
            3: text_3,
            4: text_4,
        }

    position = []
    if method == "LexRank":
        position = getLexRank(documents, language)
    if method == "NMF":
        position = getNMF(documents, language)
    if method == "TextRank":
        position = getTextRank(documents, language)
    if method == "MMR":
        position = getMMR(documents, language)
    if method == "KmeanPMMR":
        text_1_token = text_1.strip().lower()
        text_1_token = ViTokenizer.tokenize(text_1_token).split()

        text_2_token = text_2.strip().lower()
        text_2_token = ViTokenizer.tokenize(text_2_token).split()
        length_summary = (len(text_1_token) + len(text_2_token)) // 2
        position = getKmeanPMMR(documents, language, length_summary)

    j = 1
    for [pos_sen, pos_doc] in position:
        array_document[int(pos_doc) + 1][pos_sen][0] = j
        j += 1
    result['document'] = array_document
    return jsonify(result)


@app.route('/api/v1/resources/result_topic', methods=['GET'])
def api_get_result_topic():
    if ('method' in request.args) and ('topic_id' in request.args):
        topic_id = int(request.args['topic_id'])
        method = str(request.args['method'])
        language = str(request.args['language'])
        if topic_id == 178:
            return "Error: No topic_id field provided. Please specify an topic_id."
    else:
        return "Error: No topic_id field provided. Please specify an topic_id."
    if language == "en":
        list_method = {
            'MMR': {
                'folder_method': "MMR_results",
                'tail_file': "MMR",
            },
            'LexRank': {
                'folder_method': "LexRank_results",
                'tail_file': "LexRank",
            },
            'TextRank': {
                'folder_method': "TextRank_results",
                'tail_file': "TextRank",
            },
            'NMF': {
                'folder_method': "NMF_results",
                'tail_file': "NMF",
            },
        }

        method = list_method[method]

        result_method_path = os.path.dirname(os.getcwd()) + "/Data/DUC_2007/" + method['folder_method']
        for file in os.listdir(result_method_path):
            if re.search(r"^D07" + "%02d" % topic_id, file):
                text = open(result_method_path + "/" + file, "rt", encoding="utf-8").read()
                return jsonify(text)
    elif language == "vn":
        list_method = {
            'MMR': {
                'folder_method': "MMR_results",
                'tail_file': "MMR",
            },
            'LexRank': {
                'folder_method': "LexRank_results",
                'tail_file': "LexRank",
            },
            'TextRank': {
                'folder_method': "TextRank_results",
                'tail_file': "TextRank",
            },
            'NMF': {
                'folder_method': "NMF_results",
                'tail_file': "NMF",
            },
            'KmeanPMMR': {
                'folder_method': "K_mean_results_Position_MMR",
                'tail_file': "kmean",
            },
        }
        method = list_method[method]

        result_method_path = os.path.dirname(os.getcwd()) + "/Data/VietNamese/" + method['folder_method'] + "/cluster_" + str(topic_id) + "." + \
                             method[
                                 'tail_file']
        text = open(result_method_path, "rt", encoding="utf-8").read()
        return jsonify(text)
    else:
        return None


@app.route('/api/v1/resources/result_custom', methods=['GET', 'POST'])
def api_get_result_custom():
    if ('method' in request.args) and ('length' in request.args):
        method = str(request.args['method'])
        length_summary = int(request.args['length'])
        text = request.json
    else:
        return "Error: "

    documents = [v for k, v in text['documents'].items()]

    # length_summary = 100

    if method == "TextRank":
        return jsonify(TextRank.getTextRank(documents, length_summary))

    # if method == "NMF":
    #     return jsonify(nmf_summarizer.getNMF_W2V(documents, length_summary))

    if method == "LexRank":
        return jsonify(LexRank.getLexRank(documents, length_summary))

    if method == "MMR":
        return jsonify(mmr_summarizer.getMMR(documents, length_summary))

    if method == "KmeanPMMR":
        return jsonify(mmr_summarizer.getMMR(documents, length_summary))


@app.route('/api/v1/resources/rouge_custom', methods=['GET', 'POST'])
def api_get_rouge():
    text = request.json
    system = text['system']
    human = [v for k, v in text['human'].items()]
    language = text['language']
    if language == "vn":
        rouge_1 = getRouge(1, system, human)
        rouge_2 = getRouge(2, system, human)
        return jsonify({
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
        })
    elif language == "en":
        rouge_1 = getRougeEnglish(1, system, human)
        rouge_2 = getRougeEnglish(2, system, human)
        return jsonify({
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
        })
    else:
        return jsonify({
            'rouge_1': "",
            'rouge_2': "",
        })


@app.route('/api/v1/resources/rouge_average', methods=['GET'])
def api_get_rouge_average():
    if ('method' in request.args) and ('type' in request.args) and ('id' in request.args):
        method = str(request.args['method'])
        type = str(request.args['type'])
        id = int(request.args['id'])
        language = str(request.args['language'])
    else:
        return "Error: "
    if language == "vn":
        return jsonify(getRougeAverage(method, type, id))
    elif language == "en":
        return jsonify(getRougeAverageEnglish(method, type, id))
    else:
        return None

def processFile__English(file_path_and_name):
    try:
        f = open(file_path_and_name, 'r')
        text_0 = f.read()

        # code 2007
        text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
        text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
        text_1 = re.sub("\n</TEXT>", "", text_1)

        text_1 = re.sub("<P>", "", text_1)
        text_1 = re.sub("</P>", "", text_1)
        text_1 = re.sub("\n", " ", text_1)
        text_1 = re.sub("\"", "\"", text_1)
        text_1 = re.sub("''", "\"", text_1)
        text_1 = re.sub("``", "\"", text_1)
        text_1 = re.sub(" +", " ", text_1)

        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        lines = sent_tokenizer.tokenize(text_1.strip())
        result = []
        k = 1
        for sent in lines:
            OG_sent = sent[:]
            sent = sent.strip().lower()
            line = nltk.word_tokenize(sent)

            stemmed_sentence = [porter.stem(word) for word in line]
            stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'"
                                                     and x != '!' and x != '''"''' and x != "''" and x != "'s"
                                                     and x != '_' and x != '--' and x != "(" and x != ")" and x != ";",
                                           stemmed_sentence))
            if (len(stemmed_sentence) <= 4):
                break

            if stemmed_sentence:
                result.append(OG_sent)
        return result
    except Exception:
        print("Error")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
