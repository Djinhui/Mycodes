# 用n个2*1的小矩形横或竖地无重叠的覆盖一个2*n的大矩形，总共有多少种方法
class Solution:
    def rectCover(self,number):
        if number == 0:
            return 0
        if number == 1:
            return 1
        if number == 2:
            return 2
        a,b = 1,2
        for i range(3,number+1):
            a,b = b,a+b 
        return b

