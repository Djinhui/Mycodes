# 给出n个数字 a_1,...,a_n，问最多有多少不重叠的非空区间，使得每个区间内数字的xor都等于0
n = int(raw_input().strip())
a = map(int, raw_input().split())
num = set([0])
last = 0
ans = 0
for i in a:
    last ^= i
    if last in num:
        ans += 1
        num = set([0])
        last = 0
    else:
        num.add(last)
print ans