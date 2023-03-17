# 朴素串匹配算法
def navie_match(t,p):
    # 在t中查找p
    m,n = len(p),len(t)
    i,j = 0,0 
    while i < m and j < n:
        if p[i] == t[j]:
            i,j = i+1,j+1
        else:
            i,j = 0,j-i+1
    if i==m:  #找到，返回起始下标
        return j-i 
    return -1

