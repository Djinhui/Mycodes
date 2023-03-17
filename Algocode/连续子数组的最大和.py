class Solution:
    def findGreatestSumOfSubArray(self,array):
        maxNum = None
        tmpNum = 0
        for i in array:
            if maxNum is None:
                maxNum = i
            if tmpNum + i < i:
                tmpNum = i     
            else:
                tmpNum += i
            if maxNum < tmpNum:
                maxNum = tmpNum
        return maxNum

'''
dp[i]表示以元素array[i]结尾的最大连续子数组和.
以[-2,-3,4,-1,-2,1,5,-3]为例
可以发现,
dp[0] = -2
dp[1] = -3
dp[2] = 4
dp[3] = 3
以此类推,会发现
dp[i] = max{dp[i-1]+array[i],array[i]}.
'''        

# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        n = len(array)
        dp = [ i for i in array]
        for i in range(1,n):
            dp[i] = max(dp[i-1]+array[i],array[i])
         
        return max(dp)
