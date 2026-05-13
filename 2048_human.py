from numpy.lib.function_base import blackman
import pygame
import numpy as np
import random
from numpy import array as Vec
from pygame import Vector2, draw

import copy

from model import DQN

pygame.init()

Board_size=4

PixelPerBlock = 100

screen = pygame.display.set_mode(
    (Board_size*PixelPerBlock, Board_size*PixelPerBlock), pygame.DOUBLEBUF)

clock = pygame.time.Clock()

framerate = 10

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
                    self.scored+=temp
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
            self.scored+=1
                
            
    def out_of_board(self,pos):
        if pos[0]<0 or pos[0]>=Board_size or pos[1]<0 or pos[1]>=Board_size:
            return True
        return False   
    
    def get_state(self):
        return copy.deepcopy(self.board)
    
def main_loop():
    Game=Game_2048()
    playing=True
    #agent=DQN(100,Board_size**2,4)
    while playing:
        clock.tick(framerate)
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing=False
            if event.type==pygame.KEYDOWN:
                if event.key in KEY2DIR:
                    Game.move(KEY2DIR[event.key])
        Game.draw()
        
        pygame.display.flip()
    
main_loop()
        
        
        

