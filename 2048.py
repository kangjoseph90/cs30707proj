from numpy.lib.function_base import blackman
import pygame
import numpy as np
import random
from numpy import array as Vec
from pygame import Vector2, draw
import matplotlib.pyplot as plt
import copy

from model import DQN
import torch
import math

pygame.init()

Board_size=4

PixelPerBlock = 100

screen = pygame.display.set_mode(
    (Board_size*PixelPerBlock, Board_size*PixelPerBlock), pygame.DOUBLEBUF)

pygame.display.set_caption("2048 Game in RL")

clock = pygame.time.Clock()

framerate = 100

font=pygame.font.SysFont("consolas",60,True,False)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

delta = [ #방향당 위치 변화값
    Vec([1, 0]),  # Right
    Vec([0, 1]),  # Down
    Vec([-1, 0]),  # Left
    Vec([0, -1])  # Up
]

KEY2DIR = { 
    pygame.K_UP: 3,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 0
}

def draw_block(pos,number,textcolor,blockcolor):
    block = pygame.Rect(pos[0]*PixelPerBlock+1, pos[1]*PixelPerBlock+1, PixelPerBlock-2, PixelPerBlock-2)
    pygame.draw.rect(screen, blockcolor, block)
    text=font.render(f"{number}",True,textcolor)
    screen.blit(text,(pos[0]*PixelPerBlock+PixelPerBlock/3,pos[1]*PixelPerBlock+PixelPerBlock/4))
    
def getDir(dir): #정면방향 기준 [좌측, 정면, 우측] (절대방향) 리턴
    foward = dir
    right = (dir+1) % 4
    left = (dir-1) % 4
    return [left, foward, right]

class Game_2048:
    def __init__(self):
        self.board=np.zeros((Board_size,Board_size),dtype=np.int)
        self.total_score=0
        self.scored=0
        self.start={
            0:Vec([Board_size-1,0]),
            1:Vec([Board_size-1,Board_size-1]),
            2:Vec([0,Board_size-1]),
            3:Vec([0,0])
        }
        for _ in [0,1]: self.rand()
    
    def draw(self):
        for x in range(Board_size):
            for y in range(Board_size):
                if self.board[x][y]>0:
                    draw_block((x,y),self.board[x][y],BLACK,GREEN)
        
    def rand(self):
        temp=random.randrange(0,Board_size**2)
        pos=Vec([temp%Board_size,temp//Board_size])
        if self.board[pos[0]][pos[1]]>0:
            self.rand()
            return
        self.board[pos[0]][pos[1]]=2
    
    def push(self,now,dir):
        pos=copy.deepcopy(now)
        temp=copy.deepcopy(self.board[pos[0]][pos[1]])
        if temp==0: return False
        self.board[pos[0]][pos[1]]=0
        moved=False
        while True:
            pos+=delta[dir]
            if self.out_of_board(pos):
                pos-=delta[dir]
                self.board[pos[0]][pos[1]]=temp
                return moved
            if self.board[pos[0]][pos[1]]>0:
                if self.board[pos[0]][pos[1]]==temp:
                    self.scored+=20
                    self.board[pos[0]][pos[1]]+=temp+1e6
                    moved=True
                else:
                    pos-=delta[dir]
                    self.board[pos[0]][pos[1]]=temp
                return moved
            moved=True
                
    def move(self,dir): 
        now=copy.deepcopy(self.start[dir])
        dirs=getDir(dir)
        moved=False
        for _ in range(Board_size):
            for _ in range(Board_size):
                temp=self.push(now,dir)
                moved=temp or moved
                now+=delta[dirs[2]]
            now+=delta[dirs[0]]*Board_size-delta[dirs[1]]
        for x in range(Board_size):
            for y in range(Board_size):
                if self.board[x][y]>1e6:
                    self.board[x][y]-=1e6
        if moved: 
            self.rand()
            self.scored+=10
            self.total_score+=self.scored
            return True
        return False

            
    def out_of_board(self,pos):
        if pos[0]<0 or pos[0]>=Board_size or pos[1]<0 or pos[1]>=Board_size:
            return True
        return False   
    
    def get_state(self):
        grid=np.zeros((Board_size,Board_size,Board_size*Board_size))
        for x in range(Board_size):
            for y in range(Board_size):
                if self.board[x][y]==0: continue
                grid[x][y][int(round(math.log(self.board[x][y],2)))]=1
        return torch.FloatTensor(grid.ravel())
    
    def get_reward(self):
        if self.is_dead():
            return -1000
        temp=copy.deepcopy(self.scored)
        self.scored=0
        return temp
    
    def is_dead(self):
        for x in range(Board_size):
            for y in range(Board_size):
                for dir in delta:
                    if self.out_of_board((x+dir[0],y+dir[1])):
                        continue
                    if self.board[x+dir[0]][y+dir[1]]==0 or self.board[x+dir[0]][y+dir[1]]==self.board[x][y]:
                        return False
        return True
                    
    
def main_loop(episode):
    Game=Game_2048()
    agent=DQN(episode,Board_size**4,4)
    score_history=[]
    for i in range(episode):
        playing=True
        print(agent.epsilon_threshold)
        while playing:
            clock.tick(framerate)
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing=False
            state=Game.get_state()
            action = agent.select_action(state)
            while True:
                if Game.move(int(action)): break
                action[0][0]=random.randrange(0,4)
            next_state=Game.get_state()
            reward=Game.get_reward()
            agent.memorize(state, action, reward, next_state)
            agent.optimize_model()
            Game.draw()
            pygame.display.flip()
            if Game.is_dead():
                playing=False
                print(f"episode {i+1} : {Game.total_score}")
                score_history.append(Game.total_score)
                agent.decay_epsilon()
                Game.__init__()
    plt.plot(score_history)
    plt.title(f"Result of 2048 Game in RL that {episode} times learning")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.show()

    
main_loop(500)
        
        
        

