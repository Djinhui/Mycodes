# 输入一个链表，输出该链表中倒数第k个结点。
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        firsrPointer = head
        seconderPointer = head
        for i in range(k):
            if firsrPointer == None:
                return None
            firsrPointer = firsrPointer.next
        while firsrPointer != None:
            firsrPointer = firsrPointer.next
            seconderPointer = seconderPointer.next
        return seconderPointer