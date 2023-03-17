# 一 ：单链表
class SingleNode(object):
    '''单链表的节点'''
    def __init__(self,item):
        self.item = item
        self.next = None

class SingleLinkList(object):
    def __init__(self,node=None):
        if node is not None:
            headNode = SingleNode(node)
            self.__head = headNode
        else: # 实例化时未传入参数
            self.__head = node

    def isEmpty(self):
        return self.__head is None

    def length(self):
        cur = self.__head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        cur = self.__head
        while cur is not None:
            print(cur.item)
            cur = cur.next
        print('')

    def add(self,item): # 头部添加
        node = SingleNode(item)
        # 将新节点的链接域 next 指向头节点， 即_head 指向的位置
        node.next = self.__head
        # 将链表的头_head 指向新节点
        self.__head = node

    def append(self,item): # 尾部添加
        node = SingleNode(item)
        if self.isEmpty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None: # 退出时cur是最后一个节点
                cur = cur.next
            #修改节点指向 最后一个节点的 next 指向 node
            cur.next = node

    def insert(self,pos,item): # 指定位置插入
        if pos <= 0:
            self.add(item)
        elif pos >= self.length() - 1:
            self.append(item)
        else:
            node = SingleNode(item)
            count = 0
            # pre 用来指向指定位置 pos 的前一个位置 pos-1， 初始从头节点开始移动到指定位置
            pre = self.__head
            while count < pos-1:
                count += 1
                pre = pre.next
            # 先将新节点 node 的 next 指向插入位置的节点
            node.next = pre.next
            # 将插入位置的前一个节点的 next 指向新节点
            pre.next = node

    def remove(self,item):
        cur = self.__head
        pre = None
        while cur is not None:
            if cur.item == item:
                # 如果第一个就是删除的节点
                if not pre:
                   # 将头指针指向头节点的后一个节点
                   self.__head = cur.next 
                else:
                    # 将删除位置前一个节点的 next 指向删除位置的后一个节点
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

    def search(self,item):
        cur = self.__head
        while cur is not None:
            if cur.item == item:
                return True
            cur = cur.next
        return False

# 二 双链表
class Node(object):
    def __init__(self,item):
        self.item = item
        self.next = None
        self.prev = None

class DLinkList(object):
    def __init__(self):
        self.__head = node

    def isEmpty(self):
        return self.__head

    def length(self):
        cur = self.__head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        cur = self.__head
        while cur is not None:
            print(cur.item)
            cur = cur.next
        
    def add(self,item):
        node = Node(item)
        if self.isEmpty():
            self.__head = node
        else:
            # 将 node 的 next 指向_head 的头节点
            node.next = self.__head
            # 将_head 的头节点的 prev 指向 node
            self.__head.prev = node
            # 将_head 指向 node
            self.__head = node

    def append(self,item):
        node = Node(item)
        if self.isEmpty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            # 将尾节点 cur 的 next 指向 node 
            cur.next = node
            # 将 node 的 prev 指向 cur
            node.prev = cur

    def search(self,item):
        cur = self.__head
        while cur is not None:
            if cur.item == item:
                return True
            cur = cur.next
        return False

    def insert(self,pos,item):
        if pos <= 0:
            self.add(item)
        elif pos >= self.length() - 1:
            sefl.append(item)
        else:
            node = Node(item)
            cur = self.__head
            count = 0
            while  count < pos-1:
                count += 1
                cur = cur.next
            # 将 node 的 prev 指向 cur
            node.prev = cur
            # 将 node 的 next 指向 cur 的下一个节点
            node.next = cur.next
            # 将 cur 的下一个节点的 prev 指向 node
            cur.next.prev = node
            # 将 cur 的 next 指向 node
            cur.next = node

    def remove(self,item):
        cur = self.__head
        while cur is not None:
            if cur.item == item:
                #判断是否是头节点
                if cur == self.__head:
                    self.__head = cur.next
                    #判断链表是否只有一个节点
                    if cur.next is not None:
                        cur.next.prev = None
                else:
                    cur.prev.next = cur.next
                    if cur.next is not None:
                        cur.next.prev = cur.prev
                    break
            else:
                cur = cur.next

# 三 栈
class Stack(object):
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return len(self.items) == 0
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)

# 四 队列 
class Queue(object):
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return len(self.items) == 0
    def enqueue(self,item):
        self.items.insert(0,item)
    def dequeue(self):
        self.items.pop()
    def size(self):
        return len(self.items)

# 五 二叉树
class Node:
    def __init__(self,item,lchild=None,rchild=None):
        self.item = item
        self.lchild = lchild
        self.rchild = rchild

def Tree(object):
    def __init__(self,root=None):
        self.root = root

    def add(self,item):
        node = Node(item)
        if self.root is None:
            self.root = node
        else:
            queue = []
            queue.append(self.root)
            # 对已有的节点进行层次遍历(广度)
            while queue:
                cur = queue.pop()
                if cur.lchild is None:
                    cur.lchild = node
                    return 
                elif cur.rchild is None:
                    cur.rchild = node
                    return
                else:#如果左右子树都不为空，加入队列继续判断
                    queue.append(cur.lchild)
                    queue.append(cur.rchild)

    def preOrder(self,root):
        if root is None:
            return
        print(root.item)
        self.preOrder(root.lchild)
        self.preOrder(root.rrchild)

    def inOrder(self,root):
        if root is None:
            return
        self.inOrder(root.lchild)
        print(root.item)
        self.inOrder(root.rchild)

    def postOrder(self,root):
        if root is None:
            return
        self.postOrder(root.lchild)
        self.postOrder(root.rchild)
        print(root.item)

    def breadth_traversal(self,root):
        if root is None:
            return
        queue = []
        queue.append(root)
        while queue:
            cur = queue.pop(0)
            print(cur.item)
            if cur.lchild is not None:
                queue.append(cur.lchild)
            if cur.rchild is not None:
                queue.append(cur.rchild)

    
            
            