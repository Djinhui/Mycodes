def binarySearch(alist,item):
    low = 0
    high = len(alist)-1
    while low <= high:
        mid = (low+high)//2
        guess = alist[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else guess < item:
            low = mid + 1
    return None



def binary_search(alist,item):
    if len(alist) == 0:
        return False

    else:
        mid = len(alist) // 2
        if alist[mid] == item:
            return True
        else:
            if item < alist[mid]:
                return binary_search(alist[:mid], item)
            else:
                return binary_search(alist[mid+1:], item)

