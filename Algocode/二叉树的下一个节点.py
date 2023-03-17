class TreeLinkNode:
    def __init__(self,x):
        self.val = x
        self.left = None # 左子树
        self.right = None # 右子树
        self.next = None # 父节点


class Solution:
    def GetNext(self,pNode):
        # 1 寻找右子树，如果存在就一直找到右子树的最左边就是下一个节点
        # 2 没有右子树，就寻找其父节点，一直找到它是父节点的左子树，打印父节点
        if pNode.right:
            tmpNode = pNode.right
            while tmpNode.left:
                tmpNode = tmpNode.left
            return tmpNode

        else:
            tmpNode = pNode
            while tmpNode.next:
                if tmpNode.next.left: == tmpNode:
                    return tmpNode.next
                tmpNode = tmpNode.next
            return None
            
                    