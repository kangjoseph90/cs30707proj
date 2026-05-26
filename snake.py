import os
import random
import copy
from collections import deque

import numpy as np
import torch

from numpy import array as Vec

BoardX = 20
BoardY = 20
PixelPerBlock = 40

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

delta = [
    Vec([1, 0]),    # 0: Right
    Vec([0, 1]),    # 1: Down
    Vec([-1, 0]),   # 2: Left
    Vec([0, -1]),   # 3: Up
]

_PYGAME = None
_screen = None
_font = None
_clock = None


def _init_pygame(headless=False):
    global _PYGAME, _screen, _font, _clock
    if _PYGAME is not None:
        return _PYGAME
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    _screen = pygame.display.set_mode(
        (BoardX * PixelPerBlock, BoardY * PixelPerBlock), pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Snake Game in RL")
    _font = pygame.font.SysFont("consolas", 30, True, False)
    _clock = pygame.time.Clock()
    _PYGAME = pygame
    return pygame


def Opposite(d):
    return (d + 2) % 4


def getDir(d):
    forward = d
    right = (d + 1) % 4
    left = (d - 1) % 4
    return [left, forward, right]


def norm(vec):
    return abs(vec[0]) + abs(vec[1])


def isEqual(vec1, vec2):
    vec2 = vec2 / norm(vec2)
    if np.array_equal(vec1, vec2):
        return True
    return False


def DrawBlock(position, color):
    pygame = _PYGAME
    block = pygame.Rect(
        position[0] * PixelPerBlock + 1,
        position[1] * PixelPerBlock + 1,
        PixelPerBlock - 2,
        PixelPerBlock - 2,
    )
    pygame.draw.rect(_screen, color, block)


score_history = [0]


class Snake:
    def __init__(self, frame_stack: int = 1):
        self.board = np.zeros((BoardX, BoardY), dtype=np.int32)
        self.body = [Vec([BoardX // 2, BoardY // 2]), Vec([BoardX // 2, BoardY // 2])]
        self.board[BoardX // 2][BoardY // 2] = 2
        self.dir = 0
        self.last_dir = 0
        self.consume = False
        self.genApple()
        self.score = 0
        self.last_distance = self.apple - self.body[0]
        # Frame-stack placeholder: deque of recent (3, H, W) grids.
        # k=1 keeps the original behavior; k>1 returns (3k, H, W) in getFullState.
        self.frame_stack = max(1, int(frame_stack))
        self._frame_buf: "deque[np.ndarray]" = deque(maxlen=self.frame_stack)

    def Draw(self, episode, mxscore):
        for position in self.body:
            DrawBlock(position, WHITE)
        DrawBlock(self.apple, RED)
        text = _font.render(
            f"Episode : {episode}  score : {self.score}  max score: {mxscore}",
            True,
            GREEN,
        )
        _screen.blit(text, (10, 10))

    def MoveSnake(self):
        head = self.body[0] + delta[self.dir]
        self.body.insert(0, head)
        if self.isOutOfBoard(head):
            return
        self.board[head[0], head[1]] += 1
        if np.array_equal(head, self.apple):
            self.consume = True
            self.score += 1
            self.genApple()
        else:
            tail = self.body.pop()
            self.board[tail[0]][tail[1]] -= 1

    def isDead(self):
        head = self.body[0]
        if self.isOutOfBoard(head):
            return True
        if self.board[head[0]][head[1]] > 1:
            return True
        return False

    def changeDir(self, dir_NEW):
        self.dir = dir_NEW

    def genApple(self):
        while True:
            temp = random.randrange(0, BoardX * BoardY)
            apple = Vec([temp % BoardX, temp // BoardX])
            if self.board[apple[0]][apple[1]] > 0:
                continue
            self.apple = apple
            self.last_distance = self.apple - self.body[0]
            return

    def getState(self):
        """Original windowed state (85-dim): 9x9 grid centered on head + 4 apple-dir bits."""
        head = self.body[0]
        dir = getDir(self.dir) + [Opposite(self.dir)]
        pos = copy.deepcopy(head) + delta[dir[0]] * 6 + delta[dir[1]] * 6
        grid = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                if self.isOutOfBoard(pos):
                    grid[i][j] = 0
                elif self.board[pos[0], pos[1]] > 0:
                    grid[i][j] = 0
                else:
                    grid[i][j] = 1
                pos += delta[dir[2]]
            pos += delta[dir[0]] * 9 - delta[dir[1]]
        appledir = [0, 0, 0, 0]
        toapple = self.apple - head
        for i in range(4):
            if np.inner(toapple, delta[dir[i]]) > 0:
                appledir[i] = 1
        return torch.FloatTensor(list(np.ravel(grid)) + appledir)

    def _currentGrid(self) -> np.ndarray:
        """Render the current board as a (3, BoardX, BoardY) float32 array."""
        grid = np.zeros((3, BoardX, BoardY), dtype=np.float32)
        for i, seg in enumerate(self.body):
            if self.isOutOfBoard(seg):
                continue
            if i == 0:
                grid[1, seg[0], seg[1]] = 1.0
            else:
                grid[0, seg[0], seg[1]] = 1.0
        if not self.isOutOfBoard(self.apple):
            grid[2, self.apple[0], self.apple[1]] = 1.0
        return grid

    def getFullState(self):
        """Full state for the conv encoder + 4-d direction one-hot.

        With frame_stack == 1 (default): grid is (3, BoardX, BoardY).
        With frame_stack == k > 1: the k most recent frames are concatenated
        along the channel axis -> (3k, BoardX, BoardY). Newest frame first.
        Before the buffer is full, missing slots are zero-padded.
        """
        cur = self._currentGrid()
        self._frame_buf.append(cur)
        if self.frame_stack == 1:
            grid = cur
        else:
            frames = list(self._frame_buf)
            while len(frames) < self.frame_stack:
                frames.insert(0, np.zeros_like(cur))
            # newest first -> reverse so index 0..2 are the most recent frame
            frames = frames[::-1]
            grid = np.concatenate(frames, axis=0)
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.dir] = 1.0
        return torch.from_numpy(grid), torch.from_numpy(dir_onehot)

    def getReward(self):
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
        return -1

    def isOutOfBoard(self, position):
        if (
            position[0] < 0
            or position[0] >= BoardX
            or position[1] < 0
            or position[1] >= BoardY
        ):
            return True
        return False


def load(agent):
    ans = input("load model? (Y/N) : ")
    if ans != "Y" and ans != "y":
        return
    NAME = input("name : ")
    agent.load(NAME)


def save(agent):
    ans = input("save model? (Y/N) : ")
    if ans != "Y" and ans != "y":
        return
    NAME = input("name : ")
    agent.save(NAME)


def train(episode, render=True, framerate=60, save_path=None, plot=True):
    """Original windowed-state DQN training. Kept for reference / baseline.

    Use train_dqn_window.py for the baseline experiment with headless support.
    """
    from model import DQN
    import matplotlib.pyplot as plt

    pygame = _init_pygame(headless=not render)
    os.system("cls" if os.name == "nt" else "clear")
    Game = Snake()
    agent = DQN(episode, 9 ** 2 + 4, 3)
    for i in range(episode):
        print(f"epsilon : {agent.epsilon_threshold}")
        while True:
            if render:
                _clock.tick(framerate)
                _screen.fill(BLACK)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return

            dirs = getDir(Game.dir)
            state = Game.getState()
            action = agent.select_action(state)
            Game.changeDir(dirs[action])
            Game.MoveSnake()
            next_state = Game.getState()
            reward = Game.getReward()
            agent.memorize(state, action, reward, next_state)
            agent.optimize_model(load_data=False)

            if Game.isDead():
                score_history.append(Game.score)
                agent.decay_epsilon()
                Game.__init__()
                break

            if render:
                Game.Draw(i + 1, max(score_history))
                pygame.display.flip()
        print(f"{i+1}/{episode} - score : {score_history[i]} / max : {max(score_history)}")
    if save_path is not None:
        agent.save(save_path)
    if plot:
        plt.plot(score_history)
        plt.title(f"Result of Snake Game in RL ({episode} episodes)")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")
        plt.show()


if __name__ == "__main__":
    train(100)
