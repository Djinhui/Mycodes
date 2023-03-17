# 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        n = 0xFFFFFFFF & n
        count = 0
        for c in str(bin(n)):
            if c == '1':
                count += 1
        return count

print(0xFFFFFFFF & -1)
print(0xFFFFFFFF&1)
print(2**32)