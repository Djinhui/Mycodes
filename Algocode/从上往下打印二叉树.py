# 从上到下，从左到右
class Solution:
    def PrintFromTopToBottom(self,root):
        if root is None:
            return []

        s = [root]
        res = []
        while s:
            tmp = s[0]
            res.append(tmp.val)

            if tmp.left:
                s.append(tmp.left)
            if tmp.right:
                s.append(tmp.right)
            
            del s[0]

        return res
        


def breadth_travel(self,root):
    if root is None:
        return
    queue = []
    queue.append(root)
    while queue:
        node = queue.pop(0)
        print(node.elem)
        if node.lchild is not None:
            queue.append(node.lchild)
        if node.rchild is not None:
            queue.append(node.rchild)
    