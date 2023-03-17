'''
Pytorch没有官方的高阶API一般需要用户自己实现训练循环、验证循环、和预测循环。

作者通过仿照tf.keras.Model的功能对Pytorch的nn.Module进行了封装

实现了 fit, validate,predict, summary 方法,相当于用户自定义高阶AP
'''

from torchkeras import Model, summary