# 朴素匹配
# target:目标串  pattern:模式串, 在target中匹配pattern， 通常len(pattern) << len(target)

def navie_matching(t, p):
    m, n = len(p), len(t)
    i, j = 0, 0
    while i < m and j < n:
        if p[i] == t[j]:
            i, j = i + 1, j + 1
        else:
            i, j = 0, j - i + 1

    if i == m:
        return j - i
    return -1


def navie_match(t, p):
    m, n = len(p), len(t)
    for i in range(n-m+1):
        subset = t[i:i+m]
        flag = True
        for j in range(m):
            if p[j] == subset[j]:
                continue
            else:
                flag = False
                break
        if flag:
            return i
    return -1


print(navie_matching('00000000000000000000001', '0001'))
print(navie_match('00000000000000000000001', '0001'))