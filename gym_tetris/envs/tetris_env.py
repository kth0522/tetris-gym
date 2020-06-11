import gym
import numpy as np
import sys
import random
import copy
import pygame
from gym import error, spaces, utils
from gym.utils import seeding

# SHAPE FORMATS

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

class Piece(object):  # *
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

def create_grid(locked_pos={}):  # *
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_pos:
                c = locked_pos[(j,i)]
                grid[i][j] = c
    return grid

def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def get_shape():
    return Piece(5, 0, random.choice(shapes))

def clear_rows(grid, locked):

    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0,0,0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j,i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)

    return inc


class TetrisEnv(gym.Env):
    metadata = {'render.modes':['human'],
                'video.frames_per_second':350}

    def __init__(self, seed=None):
        self.seed = seed
        if seed is None:
            self.seed =  random.randint(0, sys.maxsize)
        self.s_width = 800
        self.s_height = 1000
        self.play_width = 300
        self.play_height = 600
        self.block_size = 30

        self.top_left_x = (self.s_width - self.play_width) // 2
        self.top_left_y = self.s_width - self.play_height

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({"grid": spaces.MultiDiscrete([
                                             len(shape_colors) for i in range(10 * 20)])})

        self.reset()

    def step(self, action):
        # add one due to discrete action_space
        action += 1
        temp_score = 0
        if action <= 0 or action >= 6:
            raise Exception("Invalid action: {}".format(action))

        self.grid = self.create_grid(self.locked_positions)

        self.fall_time += self.clock.get_rawtime()
        self.level_time += self.clock.get_rawtime()
        self.clock.tick(self.metadata["video.frames_per_second"])

        # if self.level_time/1000 > 5:
        #     self.level_time = 0
        #     if self.level_time > 0.12:
        #         self.level_time -= 0.005

        if self.fall_time/1000 > self.fall_speed:
            self.fall_time = 0
            self.current_piece.y += 1

        if not(self.is_valid_space(self.current_piece, self.grid)) and self.current_piece.y > 0:
            self.current_piece.y -= 1
            self.change_piece = True

        if action == 1:
            self.current_piece.x -= 1
            if not(self.is_valid_space(self.current_piece, self.grid)):
                self.current_piece.x += 1
        elif action == 2:
            self.current_piece.x += 1
            if not(self.is_valid_space(self.current_piece, self.grid)):
                self.current_piece.x -= 1
        elif action == 3:
            self.current_piece.y += 1
            if not(self.is_valid_space(self.current_piece, self.grid)):
                self.current_piece.y -= 1
        elif action == 4:
            self.current_piece.rotation += 1
            if not(self.is_valid_space(self.current_piece, self.grid)):
                self.current_piece.rotation -= 1
        elif action == 5:
            pass

        shape_pos = convert_shape_format(self.current_piece)

        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                self.grid[y][x] = self.current_piece.color

        if self.change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                self.locked_positions[p] = self.current_piece.color
            self.current_piece = self.next_piece
            self.next_piece = get_shape()
            self.change_piece = False
            temp_score = clear_rows(self.grid, self.locked_positions) * 10
            self.score += temp_score
        done = self.is_over(self.locked_positions)
        state = self.grid
        reward = temp_score

        return state, reward, done, {}

    def reset(self):
        random.seed(self.seed)
        self.locked_positions = {}
        self.grid = self.create_grid(self.locked_positions)
        self.change_piece = False

        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.fall_time = 0
        self.fall_speed = 0.27
        self.level_time = 0
        self.score = 0
        self.clock = pygame.time.Clock()

        # for rendering
        self.screen = None
        self.last_grid = copy.deepcopy(self.grid)

    def create_grid(self, locked_pos={}):  # *
        grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (j, i) in locked_pos:
                    c = locked_pos[(j, i)]
                    grid[i][j] = c
        return grid

    def is_valid_space(self, shape, grid):
        accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
        accepted_pos = [j for sub in accepted_pos for j in sub]

        formatted = convert_shape_format(shape)

        for pos in formatted:
            if pos not in accepted_pos:
                if pos[1] > -1:
                    return False
        return True

    def is_over(self, positions):
        for pos in positions:
            x, y = pos
            if y < 1:
                return True

        return False

    def draw_grid(self, surface, grid):
        sx = self.top_left_x
        sy = self.top_left_y

        for i in range(len(grid)):
            pygame.draw.line(surface, (128, 128, 128), (sx, sy + i * self.block_size),
                             (sx + self.play_width, sy + i * self.block_size))
            for j in range(len(grid[i])):
                pygame.draw.line(surface, (128, 128, 128), (sx + j * self.block_size, sy),
                                 (sx + j * self.block_size, sy + self.play_height))

    def draw_window(self, surface, grid):
        surface.fill((0, 0, 0))
        # current score
        font = pygame.font.SysFont('comicsans', 30)
        label = font.render('Score: ' + str(self.score), 1, (255, 255, 255))

        sx = self.top_left_x + self.play_width + 50
        sy = self.top_left_y + self.play_height / 2 - 100

        surface.blit(label, (sx + 20, sy + 160))


        for i in range(len(grid)):
            for j in range(len(grid[i])):
                pygame.draw.rect(surface, grid[i][j],
                                 (self.top_left_x + j * self.block_size, self.top_left_y + i * self.block_size, self.block_size, self.block_size), 0)

        pygame.draw.rect(surface, (255, 0, 0), (self.top_left_x, self.top_left_y, self.play_width, self.play_height), 5)

        self.draw_grid(surface, grid)

    def draw_next_shape(self, shape, surface):
        sx = self.top_left_x + self.play_width + 50
        sy = self.top_left_y + self.play_height / 2 - 100
        format = shape.shape[shape.rotation % len(shape.shape)]

        for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    pygame.draw.rect(surface, shape.color, (sx + j * self.block_size, sy + i * self.block_size, self.block_size, self.block_size), 0)

    def render(self, mode='human', close=False):
        if mode == 'console':
            raise NotImplementedError
        elif mode == "human":
            try:
                import pygame
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using 'pip install pygame'".format(e)
                )
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode((self.s_width, self.s_height))
                    pygame.display.set_caption('Tetris')

                self.draw_window(self.screen, self.grid)
                self.draw_next_shape(self.next_piece, self.screen)
                pygame.display.update()

