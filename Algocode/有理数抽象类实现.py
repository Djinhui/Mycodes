# 用python类实现抽象数据类型

from typing import Type


class Rational:
    # _gcd 求最大公约数用于化简有理数
    # _gcd是在有理数类内定义的一个非实例方法，_gcd的计算不依赖于任何有理数类对象，
    # 但_gcd是为有理数类的实现而需要的一种辅助功能
    # _gcd是个静态方法，可从其定义的类的名字以圆点.调用，或从该类的对象以.调用

    # python约定以下划线_开头的属性名和函数名都当作内部使用的名字，不应该在类外使用
    # python对以双下划线__开头但不以__结尾的名字做了特殊处理，使得在类外不能直接用这个名字访问
    # 以__开头并且以__结尾的名字是特殊方法名，e.g. +号运算符对应__add__
    
    @staticmethod
    def _gcd(m,n):
        if n == 0:
            m, n = n, m
        while m != 0:
            m, n = n % m, m
        return n

    def __init__(self, num, den=1) -> None:
        if not isinstance(num, int) or not isinstance(den, int):
            raise TypeError
        if den == 0:
            raise ZeroDivisionError
        sign = 1 # 正负标志
        if num < 0:
            num, sign = -num, -sign
        if den < 0:
            den, sign = -den, -sign
        
        g = Rational._gcd(num, den)
        self._num = sign * (num // g)
        self._den = den // g

    def num(self):
        return self._num
    def den(self):
        return self._den

    def __add__(self, other):
        if not isinstance(other, Rational):
            raise TypeError

        den = self._den * other.den()
        num = (self._num * other.den() + self._den * other.num())
        return Rational(num, den)

    def __eq__(self, other):
        return self._num * other.den() == self._den * other.num()

    def __str__(self):
        return str(self._num) + '/' + str(self._den)