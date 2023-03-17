#定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））
# -*- coding:utf-8 -*-

# 空间换时间，维护一个minValue,动态添加最小值，长度与stack保持一致

class Solution:
    def __init__(self):
        self.stack = []
        self.minValue = [] 

    def push(self, node):
        # write code here
        self.stack.append(node)
        if self.minValue:
            if self.minValue[-1] > node:
                self.minValue.append(node)
            else:
                self.minValue.append(self.minValue[-1])
        else:
            self.minValue.append(node)

    def pop(self):
        # write code here
        if self.stack == []:
            return None
        self.minValue.pop()
        return self.stack.pop()

    def top(self):
        # write code here
        if self.stack == []:
            return None
        return self.stack[-1]

    def min(self):
        # write code here
        if self.minValue ==[]:
            return None
        return self.minValue[-1]
