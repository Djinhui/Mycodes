class Node(object):
    def __init__(self,item):
        self.item = item
        self.next = None
        self.prev = None

class DLinkList(object):
    def __init__(self,node=None):
        if node is not None:
            headnode=None(node)
            self.__head = headnode
        else:
            self.__head = node

    def is_empty(self):
        return self.__head is None

    def length(self):
        current = self.__head
        count = 0
        while current is not None:
            count += 1
            current = current.next
        return count

    def travel(self):
        current = self.__head
        while current is not None:
            print(current.item)
            current = current.next
        
    def add(self, item):
        node = Node(item)
        if self.__head is None:
            # 如果是空链表，将_head指向node

            self.__head = node
        else:
            # 将node的next指向_head的头节点
            node.next = self._head
            # 将_head的头节点的prev指向node
            self.__head.prev = node
            # 将_head 指向node
            self.__head = node

    def append(self, item):
        node = Node(item)
        if self.is_empty():
            self.__head = node
        else:
            current = self.__head
            while current.next is not None:
                current = current.next
            current.next = node
            node.prev = current

    def search(self, item):
        current = self.__head
        while current is not None:
            if current.item == item:
                return True
            current = current.next
        return False

    def insert(self,pos,item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() -1):
            sel.append(item)
        else:
            node  =Node(item)
            current = self.__head
            count = 0
            # 移动到指定位置的前一个位置
            while count < (pos - 1):
                count += 1
                current = current.next           
                   
            node.prev = current
            node.next = current.next
            current.next.prev = node
            current.next = node

    def remove(self,item):
        current = self._head
        while current is not None:
            if current.item == item:
                if current == self.__shead:
                    self.__head = current.next
                    if current.next:# 只有一个节点
                        current.next.prev = None
                else:
                    current.prev.next = current.next
                    if current.next: #判断是否为最后一个节点，最后一个节点的next为None，没有prev
                        current.next.prev = current.prev
                break
            else:
                current = current.next
                

           




