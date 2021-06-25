import numpy as np
from scipy import optimize

def longest_common_subsequence(text1, text2):
    """最长公共子序列"""
    n = len(text1)
    m = len(text2)
    maxlen = 0
    spans1 = []
    spans2 = []
    if n * m == 0:
        return spans1, spans2, maxlen

    dp = np.zeros((n+1, m+1), dtype=np.int32)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    maxlen = dp[-1][-1]

    i = n - 1
    j = m - 1
    while len(spans1) < maxlen:
        if text1[i] == text2[j]:
            spans1.append(i)
            spans2.append(j)
            i -= 1
            j -= 1
        elif dp[i+1, j] > dp[i,j+1]:
            j -= 1
        else:
            i -= 1
    spans1 = spans1[::-1]
    spans2 = spans2[::-1]
    return spans1, spans2, maxlen

def min_edit_distance(text1, text2):
    """Levenshtein distance"""
    n = len(text1)
    m = len(text2)

    # 空串情况
    if n * m == 0:
        return n + m

    dp = np.zeros((n+1, m+1), dtype=np.int32)

    def I(a, b):
        """指示函数"""
        return 1 if a != b else 0

    # 初始化边界状态
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # 计算所有dp值
    for i in range(1, n+1):
        for j in range(1, m+1):
            left = dp[i-1][j] + 1
            down = dp[i][j-1] + 1
            left_down = dp[i-1][j-1] + I(text1[i-1], text2[j-1])
            dp[i][j] = min(left, down, left_down)
    return int(dp[n][m])

def wasserstein_distance(p, q, C):
    """Wasserstein距离计算方法，
    p.shape=(m,)
    q.shape=(n,)
    C.shape=(m,n)
    p q满足归一性概率化
    """
    p = np.array(p)
    q = np.array(q)
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(C)
        A[i,:] = 1.0
        A_eq.append(A.reshape((-1,)))
    for i in range(len(q)):
        A = np.zeros_like(C)
        A[:,i] = 1.0
        A_eq.append(A.reshape((-1,)))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q], axis=0)
    C = np.array(C).reshape((-1,))
    return optimize.linprog(
        c=C,
        A_eq=A_eq[:-1],
        b_eq=b_eq[:-1],
        method="interior-point",
        options={"cholesky":False, "sym_pos":True}
    ).fun
