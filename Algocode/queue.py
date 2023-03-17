class Queue(object):
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.insert(0, item)
        # self.items.append(item)
    
    def dequeue(self):
        return self.items.pop()
        # return self.items.pop(0)
    
    def size(self):
        return len(self.items)

        