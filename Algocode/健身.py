import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-N", type = int, default = 1)

args = parser.parse_args()
N = args.N

exercises = {'5分钟办公室肩颈放松':5, 'HIIT高效燃脂初级':18, '全身拉伸放松':12, '哑铃手臂增肌训练':13, '睡前助眠拉伸':9,\
    '超燃HIIT系列':15, '小白快乐燃脂':15, '哑铃背部轰炸初级':18, 'HIIT5分钟快速热身':5, '晚间提升腹部燃烧':12,\
        '居家增肌上肢充能':19, '晚间疲劳舒缓拉伸':7, '15分钟小腹平坦训练':15, '久坐族骨盆前倾改善':10, '徒手胸肌训练':18,\
            '360全身燃脂':9, '居家增肌肩部打造':17, '核心训练入门':11, 'HIIT心肺循环进阶':14, 'HIIT心肺功能激活':15, \
                'HIIT全身燃动训练':15, '零基础心肺功能激活':16, '腰背僵硬缓解':12, '哑铃燃脂初级':16, '新人燃脂初体验':15, \
                    '胸肌训练入门':12, '全方位腰腹速燃':10, '360全身激活':11, '大体重高效燃脂':13, '居家高能全身速燃':10, \
                        '10分钟收腹激活训练':10, '4分钟Tabata燃脂初体验':4, '零基础腹肌训练':8, '周末15分钟瘦腹特训':15, \
                            '不伤膝全身燃脂04':17, '徒手胸背速燃':10,'哑铃胸肌打造初级':17, '懒人10分钟瘦肚子':10,\
                                '不伤膝全身燃脂01':17}

stars = {'哑铃瘦手臂入门':16, '午间碎片瘦腹训练':21, '八块腹肌塑造':17, '最强核心打造初级':15, '人鱼线雕刻':17}
exercises.update(stars)

choice = np.random.choice(list(exercises.keys()), size=N, replace=False)
total_min = 0
for exercise in choice:
    total_min += exercises.get(exercise, 0)

print('今日额外训练:', choice)
print(f'共用时{total_min}分钟')