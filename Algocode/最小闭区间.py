# 给定k个有序数组, 每个数组有个N个元素，找出一个最小的闭区间，使其包含每个数组中的至少一个元素。 
# 给定两个区间[a,b], [c,d]： 
# 如果 b-a < d-c，则认为[a, b]是更小的区间；
#如果 b-a == d-c，且a < c，则认为[a, b]是更小的区间

k = int(input())
n = int(input())
nums = []
for i in range(n):
    l=[int(x) for x in input().split()]
    for x in l:
        nums.append([x,i])
nums.sort()
dp = [0]*k
cnt = k
beg,idx = 0,0
res = []
while idx < k*n:
    [a1,b]=nums[idx]
    if dp[b]==0:
        cnt -= 1
    dp[b]+=1
    if cnt==0:
        while cnt==0:
            [a,b]=nums[beg]
            if dp[b]==1:
                cnt += 1
            dp[b] -= 1
            beg += 1
        res.append([a1-nums[beg-1][0],nums[beg-1][0],a1])
    idx+=1
res.sort()
print(res[0][1],res[0][2])
