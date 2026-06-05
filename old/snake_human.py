from numpy.core.defchararray import find
import pygame
import random
import numpy as np
from pygame import draw
from numpy import array as Vec
import torch
from model import DQN
import copy

WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
GREEN=(0,255,0)

BoardX = 20 #게임 사이즈
BoardY = 20

PixelPerBlock = 30

screen = pygame.display.set_mode(
    (BoardX*PixelPerBlock, BoardY*PixelPerBlock), pygame.DOUBLEBUF)

pygame.display.set_caption("Snake Game in RL")

clock = pygame.time.Clock()

framerate = 20


delta=[
    Vec([1,0]), #Right
    Vec([0,1]), #Down
    Vec([-1,0]),#Left
    Vec([0,-1]) #Up
]

KEY2DIR={
    pygame.K_UP:3,
    pygame.K_DOWN:1,
    pygame.K_LEFT:2,
    pygame.K_RIGHT:0
}

def Opposite(dir): 
    return (dir+2)%4

def getDir(dir):
    foward=dir
    right=(dir+1)%4
    left=(dir-1)%4
    return [left,foward,right]
def norm(vec):
    return abs(vec[0])+abs(vec[1])

def DrawBlock(position, color):
    block = pygame.Rect(position[0]*PixelPerBlock+1, position[1]
                        * PixelPerBlock+1, PixelPerBlock-2, PixelPerBlock-2)
    pygame.draw.rect(screen, color, block)

def isEqual(vec1, vec2): #두 벡터가 방향이 같은지 확인 
    vec2=vec2/norm(vec2)
    if np.array_equal(vec1,vec2):
        return True
    return False

class Snake:
    def __init__(self): #게임 초기상태
        self.board=np.zeros((BoardX,BoardY))
        self.body = [Vec([BoardX//2, BoardY//2]), Vec([BoardX//2, BoardY//2])]
        self.board[BoardX//2][BoardY//2]=2
        self.dir = 0
        self.last_dir = 0
        self.consume = False
        self.genApple()
        self.score = 0
        self.last_distance=self.apple-self.body[0]

    def Draw(self):
        for position in self.body:
            DrawBlock(position,WHITE)
        DrawBlock(self.apple,RED)

    def MoveSnake(self):
        head=self.body[0]+delta[self.dir]
        self.last_dir=self.dir
        self.body.insert(0,head)
        if self.isOutOfBoard(head):
            return
        self.board[head[0]][head[1]]+=1
        if np.array_equal(head,self.apple):
            self.genApple()
        else:
            tail=self.body.pop()
            self.board[tail[0]][tail[1]]-=1
        

    def isDead(self): #죽었는지 확인
        head = self.body[0]
        if self.isOutOfBoard(head):
            return True
        if self.board[head[0]][head[1]]>1:
            return True
        return False

    def changeDir(self,dir_NEW):
        if Opposite(self.last_dir) == dir_NEW or self.last_dir == dir_NEW:
            return False
        self.dir=dir_NEW
        return True

    def genApple(self): #사과 생성
        while True:
            temp = random.randrange(0, BoardX*BoardY)
            apple = Vec([temp % BoardX, temp//BoardX])
            if self.board[apple[0]][apple[1]]>0:
                continue
            self.apple = apple
            self.last_distance=self.apple-self.body[0]
            return

    
    def getState(self): #현재 state 리턴  
        head = self.body[0]
        dir = getDir(self.dir)+[Opposite(self.dir)]  # left,foward,right
        pos=copy.deepcopy(head)+delta[dir[0]]*6+delta[dir[1]]*6
        grid=np.zeros((13,13))
        for i in range(13):
            for j in range(13):
                if self.isOutOfBoard(pos):
                    grid[i][j]=0
                elif self.board[pos[0],pos[1]]>0:
                    grid[i][j]=0
                else: 
                    grid[i][j]=1
                pos+=delta[dir[2]]
            pos+=delta[dir[0]]*13-delta[dir[1]]
        appledir = [0, 0, 0, 0]
        toapple = self.apple-head
        for i in range(4):
            if np.inner(toapple, delta[dir[i]]) > 0:
                appledir[i] = 1
        return torch.FloatTensor(list(np.ravel(grid))+appledir)

    def getReward(self): #사과 먹었으면 50점 죽으면 -100점 가까워지면 5점 멀어지면 -2점
        if self.consume:
            self.consume = False
            return 50
        if self.isDead():
            return -500
        now_distance = self.apple - self.body[0]
        if norm(now_distance) < norm(self.last_distance):
            self.last_distance = now_distance
            return 3
        self.last_distance = now_distance
        return -5

    def isOutOfBoard(self,position):
        if position[0]<0 or position[0]>=BoardX or position[1]<0 or position[1]>=BoardY:
            return True
        return False



def MainLoop():
    Game=Snake()
    running=True
    last_input=[]
    agent=DQN(1,13**2+4,3)
    while running:
        clock.tick(framerate)
        screen.fill(BLACK)
        for dir in last_input:
            if Game.changeDir(dir):
                break
        last_input.clear()
        gotInput=False
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            if event.type==pygame.KEYDOWN:
                if event.key in KEY2DIR:
                    if gotInput:
                        last_input.append(KEY2DIR[event.key])
                    else:
                        temp=Game.changeDir(KEY2DIR[event.key])
                        gotInput=True or temp is True
        state=Game.getState()
        dirs=getDir(Game.last_dir)
        action=torch.FloatTensor([dirs.index(Game.dir)])
        Game.MoveSnake()
        reward=Game.getReward()
        next_state=Game.getState()
        agent.memorize(state,action,reward,next_state)
        if Game.isDead():
            Game.__init__()
        Game.Draw()
        pygame.display.flip()
    agent.save_human_data()

MainLoop()