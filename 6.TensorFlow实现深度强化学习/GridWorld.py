import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt


# 创建环境内物体类：self、xy坐标、尺寸、亮度值、RGB颜色通道、奖励值、名称
class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


# GridWorld游戏环境：self、环境尺寸
class gameEnv():
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size
        # 环境的Action Space设置为4
        self.actions = 4
        # 初始化环境物体对象列表
        self.objects = []
        # 重置环境
        a = self.reset()
        # 显示界面
        print(a)
        print(a.shape)
        plt.imshow(a, interpolation="nearest")

    # 环境重置方法
    def reset(self):
        # 物体对象列表清空
        self.objects = []
        # 1个hero用户控制对象，通道数2显示蓝色，newPosition会随机选择一个没有占用的位置生成对象
        hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        # 4个goal有益目标，reward为1，通道数1显示绿色
        goal = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)
        goal2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal2)
        goal3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal4)
        # 2个fire有害目标，reward为-1，通道数0显示红色
        hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)
        # 绘制图像
        state = self.renderEnv()
        self.state = state
        return state

    # 控制目标移动方法
    def moveChar(self, direction):
        # 0-上、1-下、2-左、3-右
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        # 控制英雄移动发现，边界不做移动
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        self.objects[0] = hero

    # newPosition初始化物体位置的方法
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        # points初始化所有点的集合
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        # currentPositions获取所有物体位置集合
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        # 从points中去除currentPositions
        for pos in currentPositions:
            points.remove(pos)
        # 从剩余的points中随机抽取一个位置返回
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    # 检查hero是否触碰goal或fire方法
    def checkGoal(self):
        others = []
        # 从objects中获取hero对象
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                # 剩余对象存入others
                others.append(obj)
        # 遍历others列表
        for other in others:
            # 有other物体和hero重合说明有触碰发生
            if hero.x == other.x and hero.y == other.y:
                # 移除触碰物体
                self.objects.remove(other)
                # 触碰到goal，重新生成一个goal
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                # 触碰到fire，重新生成一个fire
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
                # 返回该物体的reward
                return other.reward, False
        # 未触碰返回0
        return 0.0, False

    # 获取游戏环境状态方法
    def renderEnv(self):
        # 长宽为size+2，通道数为3白色
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        # 最外面一圈的内部显示黑色
        a[1:-1, 1:-1, :] = 0
        # 遍历物体对象列表，设置初始亮度
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
        # 标准化为84*84*3尺寸的正常游戏图像尺寸
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)
        return a

    # 每一步Action方法
    def step(self, action):
        # 移动hero位置
        self.moveChar(action)
        # 检测是否有触碰，返回reward done标记
        reward, done = self.checkGoal()
        # 获取环境图像state
        state = self.renderEnv()
        # 返回state，reward，done
        return state, reward, done


env = gameEnv(size=5)
