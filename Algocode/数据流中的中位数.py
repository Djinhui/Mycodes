# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
    def Insert(self, num):
        # write code here
        self.stack.append(num)
        self.stack.sort()
    def GetMedian(self):
        # write code here
        n = len(self.stack)
        if n%2 != 0:
            return self.stack[n//2]
        else:
            return (self.stack[n//2-1]+self.stack[n//2])/2