def factorial(n):
    if n == 1:   # base condition
        return 1
    else:
        return n * factorial(n-1)
