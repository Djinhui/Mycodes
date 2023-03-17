class Node(object):
    def __init__(self,elem):
        self.elem = elem
        self.next = None

class SingleLinkList(object):
    '''单链表'''
    def __init__(self,node=None):
        if node is not None:
            headnode=None(node)
            self.__head = headnode
        else:
            self.__head = node


    def is_empty(self):
        return self._head is None

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
            print(cur.elem)
            cur = cur.next

    def add(self,elem):
        node = Node(elem)
        #顺序不能反
        node.next = self._head
        self._head = node

    def append(self,elem):
        node = Node(elem)
        if self.is_empty():
            self._head = node
        else:
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self,pos,elem):
        if pos <= 0:
            self.add(elem)
        elif pos > self.length() - 1:
            self.append(elem)
        else:
            node = Node(elem)
            count = 0
            pre = self._head
            while count < pos - 1:
                count += 1
                pre = pre.next
            node.next = pre.next
            pre.next = node


    def remove(self,elem):
        cur = self._head
        pre = None
        while cur is not None:
            if cur.elem == elem:
                if not pre:
                    self._head =cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

    def search(self,elem):
        cur = self._head
        while cur is not None:
            if cur.elem == elem:
                return True
            cur = cur.next
        return False

if __name__ == "__main__":
    ll = SingleLinkList()
    ll.add(1)
    ll.add(2)
    ll.append(3)
    ll.insert(2, 4)



