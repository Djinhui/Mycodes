# 输入两个链表，找出它们的第一个公共结点。
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        pTmp1 = pHead1
        pTmp2 = pHead2
        while pTmp1 and pTmp2:
            if pTmp1 == pTmp2:
                return pTmp1
            pTmp1 = pTmp1.next
            pTmp2 = pTmp2.next
        
        def findequal(shortPointer,longPointer,shortHead,longHead):
            k = 0
            # 寻找出两个链表的长度差
            while longPointer:
                longPointer = longPointer.next
                k += 1
            longPointer = longHead
            shortPointer = shortHead
            for i in range(k): # 先让长链表走k步
                longPointer = longPointer.next

            while longPointer != shortPointer:
                longPointer = longPointer.next
                shortPointer = shortPointer.next
            return shortPointer

        if pTmp1:
            return findequal(pTmp2,pTmp1,pHead2,pHead1)        
        if pTmp2:
            return findequal(pTmp1,pTmp2,pHead1,pHead2)
        