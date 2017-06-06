# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, jsonify
from gensim.models import KeyedVectors
from scipy.sparse import csr_matrix
from candidate_info import candidates
from math import log
import numpy as np
import ujson as json
import pickle


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def get_similar_words_(word, topn=200, num=10):
    words = model.similar_by_word(word, topn=topn)
    words = [(w, sim) for (w, sim) in words if len(w) > 1 and w not in spam]
    words = [(w, sim*log(vocaDict[w].count)) for (w, sim) in words]
    words = sorted(words, key=lambda x: x[1], reverse=True)[:num]
    words = [w[0] for w in words]
    return words


# Configuration
app = Flask(__name__, static_path='/static')

with open("article_data.pkl", 'rb') as f:
    titles = pickle.load(f)
    voca2index = pickle.load(f)
tdm = load_sparse_csr('tdm.npz')

model = KeyedVectors.load_word2vec_format('word2vec.txt')
voca = model.index2word
vocaDict = model.vocab
num_return_articles = 15

spam = set(['후보'])
for c in candidates:
    spam.update([c['name'][:2], c['name'][1:]])
    c['similar_words'] = get_similar_words_(c['name'])


@app.route('/get_related_articles')
def get_related_articles():
    axis_words = json.loads(request.args.get('words'))
    candidate = request.args.get('candidate')

    if len(axis_words) > 0:
        indices = [voca2index[w] for w in axis_words]
        indices = tdm[:, indices].sum(axis=1).argsort(axis=0)[::-1][:300]
        indices = indices.transpose().tolist()[0]
        related_titles = [titles[i] for i in indices if candidate in titles[i]]
        related_titles = related_titles[:num_return_articles]
    else:
        indices = tdm[:, voca2index[candidate]].sum(axis=1).argsort(axis=0)[::-1][:num_return_articles]
        indices = indices.transpose().tolist()[0]
        related_titles = [titles[i] for i in indices]
    return jsonify(titles=related_titles)


@app.route('/recommend_words')
def recommend_words():
    query_string = request.args.get('query')
    words = []
    for word in voca:
        if word.startswith(query_string):
            words.append(word)
            if len(words) == 5:
                break

    if query_string not in words and query_string in voca:
        words.insert(0, query_string)
        words.pop(-1)

    return jsonify(words=[{'word': word} for word in words])


@app.route('/get_similarity')
def get_word_vectors():
    axis_words = json.loads(request.args.get('words'))
    return json.dumps([{c['name']: round(model.similarity(c['name'], w), 5) for c in candidates} for w in axis_words])


@app.route('/get_similar_words')
def get_similar_words():
    word = request.args.get('word')
    return json.dumps(get_similar_words_(word))


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html', candidates=candidates)


# Execute the main program
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5015)
