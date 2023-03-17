def func(s):
    # return s.index(list(filter(lambda x:s.count(x)==1,s))[0]) if s else -1
    if not s:
        return -1
    for i in s:
        if s.count(i)==1:
            return s.index(i)
            break