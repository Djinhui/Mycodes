# 给定一棵二叉搜索树，找出其中第k小的节点
# 二叉搜索树的中序遍历是有序序列
class Solution:
    def kthNode(self,pRoot,k):
        retlist = []
        def midorder(pRoot):
            if pRoot is None:
                return None
            midorder(pRoot.left)
            retlist.append(pRoot)
            midorder(pRoot.right)
        midorder(pRoot)
        if len(retlist) < k or k < 1:
            return None
        return retlist[k-1]


