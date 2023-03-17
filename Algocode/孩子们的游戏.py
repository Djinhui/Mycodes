# 首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。
# 每次喊到m-1的那个小朋友要出列唱首歌,
# 然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
# 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,
# 可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
# 请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

# 如果没有小朋友，请返回-1
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n <1 or m <1:
            return -1
        if n==1:
            return 0
        value = 0

        for index in range(2,n+1):
            curValue = (value + m) % index
            value = curValue
        return value