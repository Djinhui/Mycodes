class Solution:
    def PrintPath(self,root,exceptNumber):
        def subFindPath(root):
            if root:
                b.append(root.val)
                if not root.right and not root.left and sum(b) == exceptNumber:
                    a.append(b[:])
                else:
                    subFindPath(root.left),subFindPath(root.right)
                b.pop()
        a, b = [], []
        subFindPath(root)
        return a  

# 遍历二叉树的所有路径
def printPath(root):
    def subFindPath(root):
        if root:
            b.append(root.val)
            if root.left is None and root.right is None:
                a.append(b)
            else:
                subFindPath(root.left)
                subFindPath(root.right)
            b.pop()
    a,b=[],[]
    subFindPath(root)
    return a
