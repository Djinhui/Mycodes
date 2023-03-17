# 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方
class Solution:
    def Power(self, base, exponent):
        # write code here
        res = 1
        if base == 0:
            return 0
        if exponent == 0:
            return 1
        if exponent < 0:
            for i in range(-exponent):
                res = base * res
            return 1 / res
        for i in range(exponent):
            res = res * base
        return res
            
    def fast_power(self,base,exponent):
        if base == 0:
            return 0
        if exponent == 0:
            return 1
        e = abs(exponent)
        tmp = base
        res = 1
        while e > 0:
            # 如果最后一位为1，那么给res乘以这一位的结果
            if e & 1 == 1:
                res = res * tmp
            e >> 1 
            tmp = tmp * tmp
        return res if exponent > 0 else 1 / res