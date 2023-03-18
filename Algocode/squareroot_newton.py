def squareroot(n):
    root = n // 2 # 初始化
    for i in range(100): # 迭代100次
        root = (1/2) * (root + (n / root))

    return root

print(squareroot(100))