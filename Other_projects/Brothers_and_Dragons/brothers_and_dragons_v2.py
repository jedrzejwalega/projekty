import pygame
from pygame.locals import *
import enum


class Move(enum.Enum):
    up = -1, 0, -2, 0
    down = 1, 0, 2, 0
    left = 0, -1, 0, -2
    right = 0, 1, 0, 2

class World():

    def __init__(self, filepath, texture_size=64):
        self.filepath = filepath
        self.texture_size = texture_size
        self.movable = None
        self.background = None
        self.player_x = None
        self.player_y = None
        self.rows = None
        self.columns = None
        self.x_pixels = None
        self.y_pixels = None
        self.surf = None
        self.non_player_positions = None
        self.banned_positions = None
        level = self.read_level()
        self.load_background(level)
        self.load_movable(level)

    def read_level(self):
        with open(self.filepath, "r") as handle:
            level = handle.read()
        return level

    def load_background(self, level):
        raw_rows = [line for line in level.split("\n") if line != ""]
        rows = []
        for line in raw_rows:
            new_line = []
            for element in line:
                if element in "#*":
                    new_line.append(element)
                else:
                    new_line.append(".")
            rows.append(new_line)
        self.background = rows
        rows_num = len(rows)
        columns_num = len(rows[0])
        self.rows = rows_num - 1
        self.columns = columns_num - 1
        self.x_pixels = rows_num * self.texture_size
        self.y_pixels = columns_num * self.texture_size
        self.surf = pygame.display.set_mode((self.y_pixels, self.x_pixels))

    def load_movable(self, level):
        raw_rows = [line for line in level.split("\n") if line != ""]
        rows = []
        
        x = 0
        y = 0

        for line in raw_rows:
            new_line = []
            for element in line:
                if element == "X":
                    new_line.append(element)
                    y += 1
                elif element == "O":
                    self.player_x = x
                    self.player_y = y
                    new_line.append(element)
                    y += 1
                elif element not in "XO":
                    new_line.append(".")
                    y += 1
            x += 1
            y = 0

            rows.append(new_line)
        self.movable = rows

    def move_player(self, pressed_key):
        if pressed_key == None:
            return None

        self.non_player_positions = self.banned_positions.union(self.exit_positions)
        player_x_moved = self.player_x + pressed_key.value[0]
        player_y_moved = self.player_y + pressed_key.value[1]

        if self.movable[player_x_moved][player_y_moved] != "X" and self.can_move(dir=pressed_key, x=self.player_x, y=self.player_y, banned_positions=self.banned_positions):
            self.movable[self.player_x][self.player_y] = "."
            self.movable[player_x_moved][player_y_moved] = "O"
            self.player_x = player_x_moved
            self.player_y = player_y_moved

        elif self.movable[player_x_moved][player_y_moved] == "X" and self.can_move(dir=pressed_key, x=player_x_moved, y=player_y_moved, banned_positions=self.non_player_positions):
            crate_x_moved = self.player_x + pressed_key.value[2]
            crate_y_moved = self.player_y + pressed_key.value[3]

            self.movable[crate_x_moved][crate_y_moved] = "X"
            self.movable[player_x_moved][player_y_moved] = "O"
            self.movable[self.player_x][self.player_y] = "."
            self.player_x = player_x_moved
            self.player_y = player_y_moved


    def can_move(self, dir, x, y, banned_positions):
        if dir == Move.up:
            if x > 0 and (x-1 , y) not in banned_positions:
                return True
        elif dir == Move.down:
            if x < self.rows and (x+1, y) not in banned_positions:
                return True
        elif dir == Move.left:
            if y > 0 and (x, y-1) not in banned_positions:
                return True
        elif dir == Move.right:
            if y < self.columns and (x, y+1) not in banned_positions:
                return True
        return False
    
    def paint_background(self):
        x = 0
        y = 0
        
        self.banned_positions = set()
        self.exit_positions = set()
        for map_row in self.background:
            for element in map_row:
                if element == ".":
                    y += self.texture_size
                    continue
                if element == "#":
                    wall = pygame.image.load("/home/jedrzej/Desktop/Scripts/Other_projects/Brothers_and_Dragons/wall.png")
                    self.surf.blit(wall, (y, x))
                    self.banned_positions.add((x // self.texture_size, y // self.texture_size))
                    y += self.texture_size
                if element == "*":
                    exit = pygame.image.load("/home/jedrzej/Desktop/Scripts/Other_projects/Brothers_and_Dragons/exit.png")
                    self.surf.blit(exit, (y, x))
                    self.exit_positions.add((x // self.texture_size, y // self.texture_size))
                    y += self.texture_size
            x += self.texture_size
            y = 0
        
    def paint_movable(self):
        x = 0
        y = 0
        
        for map_row in self.movable:
            for element in map_row:
                if element == ".":
                    y += self.texture_size
                    continue
                if element == "O":
                    player = pygame.image.load("/home/jedrzej/Desktop/Scripts/Other_projects/Brothers_and_Dragons/pacman.png")
                    self.surf.blit(player, (y, x))
                    y += self.texture_size
                if element == "X":
                    crate = pygame.image.load("/home/jedrzej/Desktop/Scripts/Other_projects/Brothers_and_Dragons/crate.png")
                    self.surf.blit(crate, (y, x))
                    self.banned_positions.add((x // self.texture_size, y // self.texture_size))
                    y += self.texture_size
            x += self.texture_size
            y = 0

def main(filepath:str, texture_size:int=64):
    world = World(filepath)
    clock = pygame.time.Clock()
    pressed_key = None
    running = True
    while running == True:
        world.paint_background()
        world.paint_movable()
        pygame.display.update()
        if (world.player_x, world.player_y) in world.exit_positions:
            running = False
        for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_UP:
                        pressed_key = Move.up
                    elif event.key == K_DOWN:
                        pressed_key = Move.down
                    elif event.key == K_LEFT:
                        pressed_key = Move.left
                    elif event.key == K_RIGHT:
                        pressed_key = Move.right
                if event.type == KEYUP:
                    pressed_key = None

        world.move_player(pressed_key)
        world.surf.fill((255,239,213))
        clock.tick(8)
    
    font = pygame.font.Font('freesansbold.ttf', 32) 
    text = font.render("You won!", True, (139,69,19), (205,133,63)) 
    textRect = text.get_rect()  
    textRect.center = (world.y_pixels // 2, world.x_pixels // 2) 

    world.surf.blit(text, textRect)
    pygame.display.update()

pygame.init()
main("/home/jedrzej/Desktop/level2.txt")