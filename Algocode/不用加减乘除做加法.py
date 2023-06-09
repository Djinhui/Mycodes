# 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # write code here
        xorNum = num1 ^ num2
        andNum = (num1 & num2) << 1

        while andNum != 0:
            tmp1 = xorNum ^ andNum
            tmp2 = (xorNum & andNum) << 1
            tmp1 = tmp1 & 0xffffffff
            xorNum = tmp1
            andNum = tmp2

        return xorNum if xorNum <= 0x7fffffff else xorNum - 0x100000000 # 用了减号
        return xorNum if xorNum <= 0x7fffffff else ~(xorNum^0xffffffff)

# 两个数相加可以看成两个数的每个位先相加，但不进位，然后在加上进位的数值
# 如12+8可以看成1+0=1 2+8=0，由于2+8有进位，所以结果就是10+10=20
# 二进制中可以表示为1000+1100 先每个位置相加不进位，
# 则0+0=0 0+1=1 1+0=1 1+1=0这个就是按位异或运算
# 对于1+1出现进位，我们可以使用按位与运算然后在将结果左移一位
# 最后将上面两步的结果相加，相加的时候依然要考虑进位的情况，直到不产生进位
# 注意python没有无符号右移操作，所以需要越界检查
# 按位与运算：相同位的两个数字都为1，则为1；若有一个不为1，则为0。
# 按位异或运算：相同位不同则为1，相同则为0。
while num2:
    result = (num1 ^ num2) & 0xffffffff
    carry = ((num1 & num2) << 1) & 0xffffffff
    num1 = result
    num2 = carry
if num1 <= 0x7fffffff:
    result = num1
else:
    result = ~(num1^0xffffffff)
return result