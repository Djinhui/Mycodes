L = [x*2 for x in range(5)]
G = (x*2 for x in range(5)) # 生成器

print(next(G))
print(next(G))
print(next(G))
print(next(G))
print(next(G))

def fib(times):
    n=0
    a,b=0,1
    while n < times:
        yield b
        a,b = b,a+b
        n += 1
    return 'DONE'
F = fib(5)
print(next(F))
print(next(F))
print(next(F))
print(next(F))
print(next(F))
for i in fib(5): # 取不到Done
    print(i)
g = fib(5)
while True:
    try:
        x=next(g)
        print('value:%d'%x)
    except StopIteration as e:
        print('value:%s'%e.value)
        break

# 闭包函数必须满足两个条件:1.函数内部定义的函数 2.包含对外部作用域而非全局作用域的引用
def outer():
    x=1 # 条件2
    def inner():
        print(x)
        print('inner func excuted')
    inner()
    print('outer func excuted')
outer()

def outer():
    x = 1
    def inner():
        print("x=%s" %x)
        print("inner func excuted")
    print("outer func excuted")
    return inner # 返回内部函数名
outer()

# 装饰器：外部函数传入被装饰函数名，内部函数返回装饰函数名。
# 特点：1.不修改被装饰函数的调用方式 2.不修改被装饰函数的源代码
# a.无参装饰器
import time,random
def outer(func):
    def inner():
        start = time.time()
        func()
        end = time.time()
        print(end - start)
    return inner

def myfunc():
    print('hello world')
outer(myfunc)()
# b.有参装饰器
def outer(func):
    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(end - start)
    return inner
# 被装饰的函数有返回值
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(end - start)
        return res
    return wrapper
def home(name):
    print('welcome to %s home page' % name)
    return 122333344445
# 等式右边返回的是wrapper的内存地址,再将其赋值给home，这里的home不在是原来的的那个函数,而是被装饰以后的函数了。
home = timer(home)
# 像home = timmer(home)这样的写法,python给我们提供了一个便捷的方式------语法糖@
# 以后我们只要在被装饰的函数之前写上@timmer,它的效果就和home = timmer(home)是一样的
# 多个装饰器装饰一个函数,其执行顺序是从下往上

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(end - start)
    return wrapper
def auth(func):
    def demo(*args, **kwargs):
        name = input('name:')
        password = input('password:')
        if name == 'dfggd' and password == '123':
            print('OK')
            func(*args, **kwargs)
        else:
            print('ERROR')
    return demo

@auth # index = auth(timmer(index)) 
@timer # index = timmer(index)
def index():
    print('welcome to index page')
index()

