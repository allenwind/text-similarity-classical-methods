import random

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC, nums=None):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines[:nums]:
        x1, x2, label = line.strip().split("\t")
        if len(x1) * len(x2) == 0:
            continue
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

_ATEC = "/home/zhiwen/workspace/dataset/matching/ATEC/totals.txt"
def load_ATEC(file=_ATEC, shuffle=True):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    if shuffle:
        random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        _id, x1, x2, label = line.strip().split("\t")
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

_BQ = "/home/zhiwen/workspace/dataset/matching/BQ_corpus/totals.txt"
def load_bq_corpus(file=_BQ, shuffle=True):
    # http://icrc.hitsz.edu.cn/Article/show/175.html
    # https://www.aclweb.org/anthology/D18-1536.pdf
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    if shuffle:
        random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        try:
            x1, x2, label = line.strip().split(",")
        except ValueError:
            # 跳过591个坏样本
            continue
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

if __name__ == "__main__":
    X1, X2, *_ = load_lcqmc()
    sl = [len(i) for i in (X1 + X2)]
    slavg = sum(sl) / len(sl)
    print(slavg)
