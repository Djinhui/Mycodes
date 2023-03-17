#对称二叉树 ： 二叉树 = 其镜像
# 镜像
'''
class Solution:
    def Mirror(self,root):
        if root is None:
            return None
        
        root.left,root.right = root.right,root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
'''
class Solution:
    def isSymmetric(self,pRoot):
        def isMirror(left,right):
            if left is None and right is None:
                return True
            elif left is None or right is None:
                return False
            if left.val != right.val:
                return False
            ret1 = isMirror(left.left,right.right)
            ret2 = isMirror(left.right,right.left)
            return ret1 and ret2

        if pRoot is None:
            return True
        return isMirror(pRoot.left,pRoot.right)  
