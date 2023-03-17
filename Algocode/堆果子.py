# N堆果子，第i堆坐标为(Xi,Yi),重量Wi,把所有果子合并为一堆，将第i堆合并到第j堆消耗Wi*(|Xi-Xj|+|Yi-Yj|)的体力，
# 求最少总体力
N = int(input())
X = [0 for _ in range(N)]
Y = [0 for _ in range(N)]
W = [0 for _ in range(N)]
w2,wx,wy = 0.0,0.0,0.0
for i in range(N):
    X[i],Y[i],W[i] = map(int,input().split())
    wx += X[i] * W[i]
    wy += Y[i] * W[i]
    w2 += W[i]
x0,y0 = wx//2,wy//2

dis = []
pos = 0
t = 0.0
for i in range(N):
    temp = abs(X[i]-x0)+abs(Y[i]-y0)
    if i == 0:
        t=temp
        pos = 0
    elif t>temp:
        t=temp
        pos = i
dis.append(pos)

for i in range(N):
    tmp = abs(X[i]-x0)+abs(Y[i]-y0)
    if tmp < t+2:
        if len(dis) > 10:
            break
        dis.append(i)
ans = 0
for i in range(len(dis)):
    cur = 0
    for j in range(N):
        if j == dis[i]:
            continue
        cur += (abs(x[j]-x[dis[i]]) + abs(y[j]-y[dis[i]])) * W[j]
    if ans == 0:
        ans = cur
    else:
        ans = min(cur,ans)
print(ans)


