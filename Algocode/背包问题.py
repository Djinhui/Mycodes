#对于给定的一个金额，需要判断能不能用不同种产品（一种产品在礼包最多出现一次）组合出来这个金额。
# 动态规划，背包问题
# N = 6 # 6个物品
# prices = [99,199,1999,10000,30,1499] # 6个物品的价格
# M = 10238 # 金额


def dp_solve(prices,money):
    dp = [0 for _ in range(money+1)]
    dp[0] = 1
    for i in range(len(prices)):
        for j in range(money,prices[i]-1,-1):
            if dp[j] < dp[j-prices[i]]:
                dp[j] = dp[j-prices[i]]
    return dp[-1]

# 给定整数n，取若干个1到n的整数可求和等于整数m，编程求出所有组合的个数。
# 比如当n=6，m=8时，有四种组合：[2,6], [3,5], [1,2,5], [1,3,4]。限定n和m小于120
n, m = map(int, raw_input().split())
dp = [0]*(m+1)
dp[0] = 1
for i in range(1, n+1):
    for j in range(m, i-1, -1):
        dp[j] = dp[j] + dp[j-i]
print dp[-1]