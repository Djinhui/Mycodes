import numpy as np
import torch

torch.manual_seed(0)
torch.zeros((3,3))
torch.randn(9)

i = torch.tensor(1);print(i,i.dtype)
x = torch.tensor(2.0);print(x,x.dtype)
b = torch.tensor(True);print(b,b.dtype)

# 指定数据类型
i = torch.tensor(1,dtype = torch.int32);print(i,i.dtype)
x = torch.tensor(2.0,dtype = torch.double);print(x,x.dtype)

# 使用特定类型构造函数
i = torch.IntTensor(1);print(i,i.dtype)
x = torch.Tensor(np.array(2.0));print(x,x.dtype) #等价于torch.FloatTensor
b = torch.BoolTensor(np.array([1,0,2,0])); print(b,b.dtype)

# 不同类型进行转换
i = torch.tensor(1); print(i,i.dtype)
x = i.float(); print(x,x.dtype) #调用 float方法转换成浮点类型
y = i.type(torch.float); print(y,y.dtype) #使用type函数转换成浮点类型
z = i.type_as(x);print(z,z.dtype) #使用type_as方法转换成某个Tensor相同类型

# 使用view改变张量尺寸
vector = torch.arange(0,12)
print(vector)
print(vector.shape)

matrix34 = vector.view(3,4)
print(matrix34)
print(matrix34.shape)

matrix43 = vector.view(4,-1) #-1表示该位置长度由程序自动推断
print(matrix43)
print(matrix43.shape)

# 有些操作会让张量存储结构扭曲，直接使用view会失败，可以用reshape方法
matrix26 = torch.arange(0,12).view(2,6)
print(matrix26)
print(matrix26.shape)

# 转置操作让张量存储结构扭曲
matrix62 = matrix26.t()
print(matrix62.is_contiguous())

# 直接使用view方法会失败，可以使用reshape方法
#matrix34 = matrix62.view(3,4) #error!
matrix34 = matrix62.reshape(3,4) #等价于matrix34 = matrix62.contiguous().view(3,4)
print(matrix34)

#torch.from_numpy函数从numpy数组得到Tensor
arr = np.zeros(3)
tensor = torch.from_numpy(arr)
print("before add 1:")
print(arr)
print(tensor)

print("\nafter add 1:")
np.add(arr,1, out = arr) #给 arr增加1，tensor也随之改变
print(arr)
print(tensor)


# numpy方法从Tensor得到numpy数组
tensor = torch.zeros(3)
arr = tensor.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("after add 1:")
#使用带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr也随之改变 
#或： torch.add(tensor,1,out = tensor)
print(tensor)
print(arr)


# 可以用clone() 方法拷贝张量，中断这种关联
tensor = torch.zeros(3)

#使用clone方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy() # 也可以使用tensor.data.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("after add 1:")
#使用 带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr不再随之改变
print(tensor)
print(arr)


# item方法和tolist方法可以将张量转换成Python数值和数值列表
scalar = torch.tensor(1.0)
s = scalar.item()
print(s)
print(type(s))

tensor = torch.rand(2,2)
t = tensor.tolist()
print(t)
print(type(t))


#可以使用索引和切片修改部分元素
x = torch.tensor([[1,2],[3,4]],dtype = torch.float32,requires_grad=True)
x.data[1,:] = torch.tensor([0.0,0.0])
x

minval=0
maxval=100

# 有4个班级，每个班级10个学生，每个学生7门科目成绩
scores = torch.floor(minval + (maxval-minval)*torch.rand([4,10,7])).int()
print(scores)

#抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))

#抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = torch.index_select(torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))
                   ,dim=2,index = torch.tensor([1,3,6]))

#抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
#take将输入看成一维数组，输出和index同形状
s = torch.take(scores,torch.tensor([0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6]))

#抽取分数大于等于80分的分数（布尔索引）
#结果是1维张量
g = torch.masked_select(scores,scores>=80)

#如果分数大于60分，赋值成1，否则赋值成0
ifpass = torch.where(scores>60,torch.tensor(1),torch.tensor(0))

#将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分
torch.index_fill(scores,dim = 1,index = torch.tensor([0,5,9]),value = 100)
#等价于 scores.index_fill(dim = 1,index = torch.tensor([0,5,9]),value = 100)

#将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores,scores<60,60)
#等价于b = scores.masked_fill(scores<60,60)
b

# 维度变换
'''
torch.reshape 可以改变张量的形状。(或者调用张量的view方法)
torch.squeeze 可以减少维度。
torch.unsqueeze 可以增加维度。
torch.transpose 可以交换维度。
'''


# 张量的view方法有时候会调用失败，可以使用reshape方法。
torch.manual_seed(0)
minval,maxval = 0,255
a = (minval + (maxval-minval)*torch.rand([1,3,3,2])).int()
print(a.shape)

b = a.view([3,6])
c = torch.reshape(a, [3,6])
b.view([1,3,3,2])
torch.reshape(c, [1,3,3,2])

# 在某个维度上只有一个元素，利用torch.squeeze可以消除这个维度
torch.squeeze(a)
#在第0维插入长度为1的一个维度
torch.unsqueeze(b, axis=0)

# torch.transpose可以交换张量的维度,
# 二维的矩阵，通常会调用矩阵的转置方法 matrix.t()，等价于 torch.transpose(matrix,0,1)
data = torch.floor(minval + (maxval-minval)*torch.rand([100,256,256,4])).int()
# 转换成 Pytorch默认的图片格式 Batch,Channel,Height,Width 
# 需要交换两次
data_t = torch.transpose(torch.transpose(data, 1,2),1,3)

# 合并-分割
# torch.cat是连接，不会增加维度，而torch.stack是堆叠，会增加维度
a = torch.tensor([[1.0,2.0],[3.0,4.0]])
b = torch.tensor([[5.0,6.0],[7.0,8.0]])
c = torch.tensor([[9.0,10.0],[11.0,12.0]])

abc_cat = torch.cat([a,b,c],dim = 0)
print(abc_cat.shape) # （6，2）

abc_stack = torch.stack([a,b,c],axis = 0) #torch中dim和axis参数名可以混用
print(abc_stack.shape) # ([3, 2, 2])

torch.cat([a,b,c],axis = 1) # （2，6）
torch.stack([a,b,c],axis = 1) #(2,3,2)

# torch.split是torch.cat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割
a,b,c = torch.split(abc_cat,split_size_or_sections = 2,dim = 0) #每份2个进行分割
p,q,r = torch.split(abc_cat,split_size_or_sections =[4,1,1],dim = 0) #每份分别为[4,1,1]