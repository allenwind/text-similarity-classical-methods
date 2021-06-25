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

if __name__ == "__main__":
    X1, X2, *_ = load_lcqmc()
    sl = [len(i) for i in (X1 + X2)]
    slavg = sum(sl) / len(sl)
    print(slavg)
