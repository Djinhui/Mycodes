class Deque: 
    '''双端队列'''
    def __init__(self):
        self.items = []
    def isEmpty(self): 
        return self.items == [] 
    def addFront(self, item): 
        self.items.append(item) 
    def addRear(self, item): 
        self.items.insert(0,item) 
    def removeFront(self): 
        return self.items.pop()
    def removeRear(self):
        return self.items.pop(0) 
    def size(self): 
        return len(self.items)

def palcheker(Astring):
    chardeque = Deque()
    for ch in Astring:
        chardeque.addRear(ch)

    stillEqual = True
    while stillEqual and chardeque.size() > 1:
        first = chardeque.removeFront()
        last = chardeque.removeRear()
        if first != last:
            stillEqual = False

    return stillEqual