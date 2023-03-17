from collections import deque
graph = {}

def person_is_seller(person):
    if person[-1]=='m':
        return True

def BFS(name):
    search_deque = deque()
    search_deque += graph[name]
    searched = []

    while not search_deque.empty():
        person = search_deque.popleft()
        if person not in searched:
            if person_is_seller(person):
                print(person + 'is a seller')
                return True
            else:
                search_deque += graph[person]
                searched.append(person)
    return False