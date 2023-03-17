# 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，
# 另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。
# （注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if pHead is None:
            return None
        pTmp  = pHead
        while pTmp:
            node = RandomListNode(pTmp.label)
            node.next = pTmp.next
            pTmp.next = node
            pTmp = node.next

        pTmp =pHead
        while pTmp:
            if pTmp.random:
                pTmp.next.random = pTmp.random.next
            pTmp = pTmp.next.next

        pTmp = pHead
        newHead = pTmp.next
        pNewHead = pTmp.next
        while pTmp:
            pTmp.next = pTmp.next.next
            if pNewHead.next:
                pNewHead.next = pNewHead.next.next
                pNewHead = pNewHead.next
            pTmp = pTmp.next
        return newHead