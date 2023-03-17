#当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
# 当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.s = ''
    def FirstAppearingOnce(self):
        # write code here
        for ch in self.s:
            if self.s.count(ch) == 1:
                return ch
        return '#'
    def Insert(self, char):
        # write code here
        self.s += char