# 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1 is None:
            return pHead2
        if pHead2 is None:
            return pHead1

        newHead = pHead1 if pHead1.val < pHead2.val else pHead2

        pTmp1 = pHead1
        pTmp2 = pHead2

        if newHead == pTmp1:
            pTmp1 = pTmp1.next
        else:
            pTmp2 = pTmp2.next

        prvPointer = newHead

        while pTmp1 and pTmp2:
            if pTmp1.val < pTmp2.val:
                prvPointer.next = pTmp1
                prvPointer= pTmp1
                pTmp1 = pTmp1.next
            else:
                prvPointer.next = pTmp2
                prvPointer = pTmp2
                pTmp2 = pTmp2.next

        if pTmp1 is None:
            prvPointer.next = pTmp2
        else:
            prvPointer.next = pTmp1

        return newHead