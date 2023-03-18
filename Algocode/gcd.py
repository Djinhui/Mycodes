# 欧几里得算法求最大公约数
# 辗转相除

def gcd(m,n):
    while m % n != 0:
        oldm = m
        oldn = n
        m = oldn  # 将除数作为新的被除数
        n = oldm % oldn  # 将余数作为新的除数
    return n

print(gcd(1112, 695))