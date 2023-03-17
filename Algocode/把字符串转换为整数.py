#将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
# -*- coding: utf-8 -*-
class Solution:
    def StrToInt(self,s):
        if s is None:
            return 0
        lens = len(s)
        if lens == 0:
            return 0
        # numlist=['0','1','2','3','4','5','6','7','8','9','+','-']
        res = 0
        start = 0
        flag = 1
        if s[0] == '+' or s[0] == '-':
            start = 1
            if s[0] == '-':
                flag = -1
        for i in range(start,lens):
            if '0' <= s[i] <= '9':
                res = res * 10 + (ord(s[i]) - ord('0'))
                # res=res*10+numlist.index(s[i])
            else:
                return 0
        if flag == -1:
            return -1 * res
        return res