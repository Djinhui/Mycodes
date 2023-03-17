class Solution:
    def Convert(self,pRootOfTree):
        if pRootOfTree is None:
            return None

        def find_right(node):
            while node.right:
                node = node.right
            return node


        leftNode = self.Convert(pRootOfTree.left)
        rightNode = self.Convert(pRootOfTree.right)

        retNode = leftNode
        if leftNode:
            leftNode = find_right(leftNode)
        else:
            retNode = pRootOfTree

        pRootOfTree.left = leftNode
        pRootOfTree.right = rightNode

        if leftNode is not None:
            leftNode.right = pRootOfTree
        if rightNode is not None:
            rightNode.left = pRootOfTree


        return retNode

class Solution2:
    def Convert(self,pRootOfTree):
        if pRootOfTree is None:
            return None

        self.arr = []
        self.midTravelling(pRootOfTree)
        for i,v in enumerate(self.arr[:-1]):
            v.right = self.arr[i+1]
            self.arr[i+1].left = v
        return self.arr[0]

    def midTravelling(self,root):
        if not root:return
        self.midTravelling(root.left)
        self.arr.append(root)
        self.midTravelling(root.right)


