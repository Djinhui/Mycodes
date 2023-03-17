class None:
    def __init__(self,item):
        self.item = item
        self.next = None

class SingleCycleLinkList(object):
    '''单向循环链表'''
    def __init__(self,node=None):
        self.__head = node
        if node:
            node.next = node


    def is_empty(self):
        return self.___head is None

    def length(self):
        if self.__head is None:
            return 0
        count = 1
        curr = self.__head
        while curr.next != self.__head:
            count += 1
            curr = curr.next
        return count

    def travel(self):
        if self.is_empty():
            return
        curr = self.__head       
        while curr.next != self.__head:            
            print(curr.item)
            curr = curr.next
        print(curr.item)
    def add(self, item):
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = self.__head
        else:
            node.next =self.__head
            curr = self.__head
            while curr.next != self.__head:
                curr = curr.next
            curr.next = node
            self.__head = node

    def append(self, item):
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = self.__head
        else:
            curr = self.__head
            while curr.next != self.__head:
                curr = curr.next
            curr.next = node
            node.next = self.__head

    def

