# 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        # 一个指针每次走两步，一个指针每次走一步
        if pHead is None:
            return None

        fastPointer = pHead
        slowPointer = pHead

        while fastPointer and fastPointer.next: # 
            fastPointer = fastPointer.next.next
            slowPointer = slowPointer.next

            if fastPointer == slowPointer: # 有环
                break
        
        if fastPointer == None or fastPointer.next == None: # 没环
            return None

        # 如果slow走了L的长度，fast走2L
        # 设从起点到入口点的距离为s，slow在环里走了d，L = s+d
        # 假设slow在环里没走的长度为m,fast走的长度为2L = K(m+d) + d + s   (K倍环的长度)
        # 化简：s = m + (K-1)(m+d)
        fastPointer = pHead
        while fastPointer != slowPointer:
            fastPointer = fastPointer.next
            slowPointer = slowPointer.next
        return fastPointer


        

        


