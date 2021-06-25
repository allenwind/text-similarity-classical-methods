import collections
import numpy as np
import jieba
from jieba.analyse.tfidf import IDFLoader, DEFAULT_IDF
from sklearn import metrics
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from scipy.spatial import distance

from snippets import longest_common_subsequence
from snippets import longest_common_substring
from snippets import min_edit_distance
from snippets import wasserstein_distance

# 经典的句子匹配方法

# 可以在这里找合适的词向量
# https://github.com/Embedding/Chinese-Word-Vectors
path = "/home/zhiwen/workspace/dataset/word2vec_baike/word2vec_baike"
model = Word2Vec.load(path)
word_vectors = {w: model.wv[w] for w in model.wv.index2word}

idf_dict, median_idf = IDFLoader(DEFAULT_IDF).get_idf()

def cosine(v1, v2):
    s = 1 - distance.cosine(v1, v2)
    if np.isnan(s):
        return 0
    return s

def normalize(x):
    x = np.array(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def cosine_similar(text1, text2):
    words1 = jieba.lcut(text1)
    words2 = jieba.lcut(text2)
    v1 = np.array([word_vectors[w] for w in words1 if w in word_vectors]).sum(axis=0)
    v2 = np.array([word_vectors[w] for w in words2 if w in word_vectors]).sum(axis=0)
    return cosine(v1, v2)

def idf_weighted_sum_similar(text1, text2):
    words1 = jieba.lcut(text1)
    words2 = jieba.lcut(text2)
    # 使用idf作为权重加权平均
    v1 = np.array([word_vectors[w] * idf_dict.get(w, 1) for w in words1 if w in word_vectors]).sum(axis=0)
    v2 = np.array([word_vectors[w] * idf_dict.get(w, 1) for w in words2 if w in word_vectors]).sum(axis=0)
    return cosine(v1, v2)

def jaccard_similar(text1, text2):
    words1 = jieba.lcut(text1)
    words2 = jieba.lcut(text2)
    s1 = set(words1)
    s2 = set(words2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def tfidf_similar(text1, text2):
    words1 = jieba.lcut(text1)
    words2 = jieba.lcut(text2)
    words = list(set(words1).union(set(words2)))
    v1 = np.array([words1.count(w) * idf_dict.get(w, 1) for w in words])
    v2 = np.array([words2.count(w) * idf_dict.get(w, 1) for w in words])
    return cosine(v1, v2)

def bm25_similar(text1, text2, s_avg=10, k1=2.0, b=0.75):
    """s_avg是句子的平均长度，根据语料统计。k1,b是调节因子，根据经验调整。"""
    bm25 = 0.0
    sl = len(text2)
    for w in jieba.lcut(text1):
        w_idf = idf_dict.get(w, 1)
        bm25_ra = text2.count(w) * (k1 + 1)
        bm25_rb = text2.count(w) + k1 * (1 - b + b * sl / s_avg)
        bm25 += w_idf * bm25_ra / bm25_rb
    return bm25

def min_editdistance_similar(text1, text2):
    """根据编辑距离的归一化值作为相似性度量"""
    distance = min_edit_distance(text1, text2)
    return 1 - distance / max(len(text1), len(text2))

def word_mover_similar(text1, text2):
    """Word Mover's Distance计算方法, 
    x.shape=(m,d)
    y.shape=(n,d)
    """
    words1 = jieba.lcut(text1)
    words2 = jieba.lcut(text2)
    x = [word_vectors[w] for w in words1 if w in word_vectors]
    y = [word_vectors[w] for w in words2 if w in word_vectors]
    x = np.array(x)
    y = np.array(y)
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    C = np.sqrt(np.mean(np.square(x[:,None] - y[None,:]), axis=2))
    return 1 - wasserstein_distance(p, q, C)

def lcs_similar(text1, text2):
    """根据最长公共子序列的归一化值作为相似性度量"""
    *_, distance = longest_common_subsequence(text1, text2)
    return distance / max(len(text1), len(text2))

def lcsubstring_similar(text1, text2):
    """根据最长公共子串的归一化值作为相似性度量"""
    *_, distance = longest_common_substring(text1, text2)
    return distance / min(len(text1), len(text2))

def plot_pcs(ys_true, ys_pred, labels, show=True):
    for y_true, y_pred, label in zip(ys_true, ys_pred, labels):
        precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="balance", alpha=0.6)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.title("precision recall curve")

    if show:
        plt.show()

def plot_rocs(ys_true, ys_pred, labels, show=True):
    for y_true, y_pred, label in zip(ys_true, ys_pred, labels):
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="{} auc={:.3f}".format(label, auc))

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="random-chance", alpha=0.6)

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="lower right")

    if show:
        plt.show()

if __name__ == "__main__":
    import dataset
    funcs = [cosine_similar, idf_weighted_sum_similar, jaccard_similar, 
             tfidf_similar, bm25_similar, min_editdistance_similar,
             lcs_similar, lcsubstring_similar]

    # 注意word_mover_similar计算太慢了，这里默认不加入
    # funcs.append(word_mover_similar)
    do_normalize = True
    X1, X2, y, categoricals = dataset.load_lcqmc(nums=10000)
    ys_true = []
    ys_pred = []
    labels = []
    for func in funcs:
        labels.append(func.__name__)
        y_pred = [func(x1, x2) for x1, x2 in zip(X1, X2)]
        if do_normalize:
            y_pred = normalize(y_pred)
        ys_true.append(np.array(y))
        ys_pred.append(np.array(y_pred))
    plt.subplot(121)
    plot_pcs(ys_true, ys_pred, labels, show=False)
    plt.subplot(122)
    plot_rocs(ys_true, ys_pred, labels, show=True)
