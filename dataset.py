import random

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines[:]:
        x1, x2, label = line.strip().split("\t")
        if len(x1) * len(x2) == 0:
            continue
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals
