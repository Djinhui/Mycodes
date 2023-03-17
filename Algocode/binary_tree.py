class Node(object):
    def __init__(self,elem,lchild=None,rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild

class Tree(object):
    def __init__(self,root=None):
        self.root = root

    def add(self,elem):
        node = Node(elem)
        #如果树是空的，则对根节点赋值
        if self.root is None:
            self.root = node
        else:
            queue = []
            queue.append(self.root)
            # 对已有的节点进行层次遍历(广度)
            while queue:
                #弹出队列的第一个元素
                cur = queue.pop(0)
                if cur.lchild is None:
                    cur.lchild = node
                    return
                elif cur.rchild is None:
                    cur.rchild = node
                else: #如果左右子树都不为空，加入队列继续判断
                    queue.append(cur.lchild)
                    queue.append(cur.rchild)

    def preorder(self,root):
        if root is None:
            return
        print(root.elem)
        self.preorder(root.lchild)
        self.preorder(root.rchild)

    def inorder(self,root):
        if root is None:
            return
        self.inorder(root.lchild)
        print(root.elem)
        self.inorder(root.rchild)

    def postorder(self,root):
        if root is None:
            return
        self.postorder(root.lchild)
        self.preorder(root.rchild)
        print(root.elem)

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

                