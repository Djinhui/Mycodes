# 输入一个链表，反转链表后，输出新链表的表头。
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead == None:
            return None
        if pHead.next is None:
            return pHead
        leftpointer = pHead
        middlepointer = pHead.next
        rightpointer = middlepointer.next

        leftpointer.next = None

        while rightpointer is not None:
            middlepointer.next = leftpointer
            leftpointer = middlepointer
            middlepointer = rightpointer
            rightpointer = rightpointer.next
        middlepointer.next = leftpointer
        
        return middlepointer
    
