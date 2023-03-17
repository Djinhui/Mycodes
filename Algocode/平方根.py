def squareRoot(num):
    # 牛顿迭代f(x) = x**2 - num 
    res = num
    while abs(res**2-num) > 1e-6:
        res = (res + num/res) / 2
    print('%.4f'%res)