# 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here 
        # a xor a = 0
        # a xor b xor c = a xor c xor b
        # 1^1^2^2^3^4^4 = 3
        if len(array) < 2:
            return None
        
        twoNumXor = None
        for num  in array:
            if twoNumXor is None:
                twoNumXor = num
            else:
                twoNumXor = twoNumXor ^ num

        count = 0
        while twoNumXor % 2 == 0:
            twoNumXor = twoNumXor >> 1
            count += 1
        mask = 1 << count

        firstNum = None
        secondNum = None
        for num in array:
            if mask & num == 0:
                if firstNum is None:
                    firstNum = num
                else:
                    firstNum = firstNum ^ num
            else:
                if secondNum is None:
                    secondNum = num
                else:
                    secondNum = secondNum ^ num

        return firstNum, secondNum

# print(312^434)
# print(type(312^434)) # int
# print(bin(32))
# print(type(bin(32))) #str
count = 0
twoNumXor = 312^434
while twoNumXor % 2 == 0:
    print(twoNumXor)
    twoNumXor = twoNumXor >> 1
    count += 1
mask = 1 << count   
print(count)   
print(mask)