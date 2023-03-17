class Doublequeue(object):

    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def add_front(self, item):
        self.items.insert(0, item)

    def add_back(self, item):
        self.items.append(item)

    def remove_front(self, item):
        return self.items.pop(0)

    def remove_back(self, item):
        return self.items.pop()

    def size(self):
        return len(self.items)

        
    
    