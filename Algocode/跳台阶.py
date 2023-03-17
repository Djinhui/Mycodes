# f(n) = f(n-1) + f(n-2)

class Solution1:
    def jumpFloor(self, n):
        res = [1, 1, 2]
        while len(res) <= n:
            res.append(res[-1] + res[-2])
        return res[n]
s = Solution1()
print(s.jumpFloor(10))

class Solution2:
    def jumpFloor(self,number):
        if number < 1:
            return 0
        if number ==1:
            return 1
        if number ==2:
            return 2
        else:
            a = 1
            b = 2
            for i in range(2,number):
                a,b = b, a+b
            return b        


#一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
#跳1级，剩下n-1级，则剩下跳法是f(n-1)
#跳2级，剩下n-2级，则剩下跳法是f(n-2)
#所以f(n)=f(n-1)+f(n-2)+...+f(1)
#因为f(n-1)=f(n-2)+f(n-3)+...+f(1)
#所以f(n)=2*f(n-1)          

class Solution3:
    def jumpFloorII(self,number):
        if number<=0:
            return 0
        if number ==1:
            return 1
        else:
            return 2*self.jumpFloorII(number-1)
# 2^(n-1)  
class Solution4:
    def jumpFloorII(self,number): 
        return pow(2,number-1)  

"""
台阶问题考虑动态规划
每次仅可往上爬2的整数次幂个台阶(1、2、4、....)
当前台阶方法数 = 所有一次可到达当前台阶方法数的和
dp[n] = dp[n-1]+dp[n-2]+dp[n-4]+... ( n-t>=0,dp[0]=1 )
"""

# 已知深渊有N层台阶构成（1 <= N <= 1000)
# 为了防止溢出，可将输出对10^9 + 3取模
dp = [0] * 1001
mod = pow(10,9) + 3
dp[0] = 1
for i in range(1,1001):
    t = 1 
    while t <= i:
        dp[i] = dp[i-t]
        dp[i] %= mod
        t*=2
dp[n]    # n<=1000