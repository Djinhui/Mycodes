import torch 
import numpy as np 

a = torch.tensor([[1.0,2],[-3,4.0]])
b = torch.tensor([[5.0,6],[7.0,8.0]])
a + b  #运算符重载
a - b 
a * b  # 对应元素相乘，非矩阵乘法
a / b
a ** 2
a % 3
a // 3
a >= 3 # torch.ge(a, 2) ge:greater_equal
(a >= 2) & (a <= 3)
(a >= 2) | (a <= 3)
a==5 #torch.eq(a,5)
torch.sqrt(a)

a = torch.tensor([1.0,8.0])
b = torch.tensor([5.0,6.0])
c = torch.tensor([6.0,7.0])

d = a+b+c

print(torch.min(a,b))


x = torch.tensor([2.6,-2.7])

print(torch.round(x)) #保留整数部分，四舍五入
print(torch.floor(x)) #保留整数部分，向下归整
print(torch.ceil(x))  #保留整数部分，向上归整
print(torch.trunc(x)) #保留整数部分，向0归整
print(torch.fmod(x,2)) #作除法取余数 
print(torch.remainder(x,2)) #作除法取剩余的部分，结果恒正

# 幅值裁剪
x = torch.tensor([0.9,-0.8,100.0,-20.0,0.7])
y = torch.clamp(x, min=-1, max=1)


#统计值
a = torch.arange(1,10).float()
print(torch.sum(a))
print(torch.mean(a))
print(torch.max(a))
print(torch.min(a))
print(torch.prod(a)) #累乘
print(torch.std(a))  #标准差
print(torch.var(a))  #方差
print(torch.median(a)) #中位数

#指定维度计算统计值
b = a.view(3,3)
print(b)
print(torch.max(b,dim = 0))
print(torch.max(b,dim = 1))

#cum扫描
a = torch.arange(1,10)
print(torch.cumsum(a,0))
print(torch.cumprod(a,0))
print(torch.cummax(a,0).values)
print(torch.cummax(a,0).indices)
print(torch.cummin(a,0))

#torch.sort和torch.topk可以对张量排序
a = torch.tensor([[9,7,8],[1,3,2],[5,6,4]]).float()
print(torch.topk(a,2,dim = 0),"\n")
print(torch.topk(a,2,dim = 1),"\n")
print(torch.sort(a,dim = 1),"\n")

#矩阵乘法
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[2,0],[0,2]])
print(a @ b) # torch.matmul(a, b), torch.mm(a, b)
#矩阵转置
print(a.t())

#矩阵逆，必须为浮点类型
a = torch.tensor([[1.0,2],[3,4]])
print(torch.inverse(a))

#矩阵求trace
a = torch.tensor([[1.0,2],[3,4]])
print(torch.trace(a))

#矩阵求范数
a = torch.tensor([[1.0,2],[3,4]])
print(torch.norm(a))

#矩阵行列式
a = torch.tensor([[1.0,2],[3,4]])
print(torch.det(a))

#矩阵特征值和特征向量
a = torch.tensor([[1.0,2],[-5,4]],dtype = torch.float)
print(torch.eig(a,eigenvectors=True))

#矩阵QR分解, 将一个方阵分解为一个正交矩阵q和上三角矩阵r
#QR分解实际上是对矩阵a实施Schmidt正交化得到q
a  = torch.tensor([[1.0,2.0],[3.0,4.0]])
q,r = torch.qr(a)
print(q,"\n")
print(r,"\n")
print(q@r)

#矩阵svd分解
#svd分解可以将任意一个矩阵分解为一个正交矩阵u,一个对角阵s和一个正交矩阵v.t()的乘积
#svd常用于矩阵压缩和降维
a=torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
u,s,v = torch.svd(a)
print(u,"\n")
print(s,"\n")
print(v,"\n")

print(u@torch.diag(s)@v.t())


# 广播机制
a = torch.tensor([1,2,3])
b = torch.tensor([[0,0,0],[1,1,1],[2,2,2]])
print(b + a) 

a_broad,b_broad = torch.broadcast_tensors(a,b)
print(a_broad,"\n")
print(b_broad,"\n")
print(a_broad + b_broad) 