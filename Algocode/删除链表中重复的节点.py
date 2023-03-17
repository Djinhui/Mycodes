# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if not pHead:
            return pHead
        p1 = []
        while pHead:
            p1.append(pHead.val)
            pHead = pHead.next
        np1 = []
        for p in p1:
            if p1.count(p) == 1:
                np1.append(p)
        head = ListNode(0)
        h=head
        for p in np1:
            h.next = ListNode(p)
            h = h.next
        return head.next

def deleteDuplication(pHead):
    if not pHead or not pHead.next:
        return pHead
    new_head = ListNode(0)
    new_head.next = pHead
    pre = new_head
    p = pHead
    while p and p.next:
        nextNode = p.next
        if p.val == nextNode.val:
            while nextNode and nextNode.val == p.val:
                nextNode = nextNode.next
            p = nextNode
            pre.next = nextNode
        else:
            pre = p
            p = nextNode        

    return new_head.next