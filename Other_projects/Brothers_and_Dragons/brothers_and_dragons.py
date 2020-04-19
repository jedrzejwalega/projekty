import pygame
from pygame.locals import *

class World():

    def __init__(self, filepath, texture_size=64):
        self.filepath = filepath
        self.texture_size = texture_size
    
    def load_background(self):
        with open(self.filepath, "r") as handle:
            level = handle.read()
        
        raw_rows = [line for line in level.split("\n") if line != ""]
        rows = [element if element in "#*" else "." for line in level.split("\n") for element in line]
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
        

    def load_movable(self):
        with open(self.filepath, "r") as handle:
            level = handle.read()
        
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
                if element == "O":
                    self.player_x = x
                    self.player_y = y
                    new_line.append(element)
                    y += 1
                if element not in "XO":
                    new_line.append(".")
                    y += 1
            x += 1
            y = 0

            rows.append(new_line)
        self.movable = rows
    
    def move_player(self, pressed_keys):
        self.non_player_positions = self.banned_positions.union(self.exit_positions)
        
        if pressed_keys[K_UP] and self.player_x > 0:
            if self.movable[self.player_x - 1][self.player_y] != "X" and self.can_move(dir="up", x=self.player_x, y=self.player_y, banned_positions=self.banned_positions):
                self.movable[self.player_x][self.player_y] = "."
                self.movable[self.player_x - 1][self.player_y] = "O"
                self.player_x -= 1
            elif self.movable[self.player_x - 1][self.player_y] == "X" and self.can_move(dir="up", x=self.player_x - 1, y=self.player_y, banned_positions=self.non_player_positions):
                self.movable[self.player_x - 2][self.player_y] = "X"
                self.movable[self.player_x - 1][self.player_y] = "O"
                self.movable[self.player_x][self.player_y] = "."
                self.player_x -= 1

        if pressed_keys[K_DOWN] and self.player_x < self.rows:
            if self.movable[self.player_x + 1][self.player_y] != "X" and self.can_move(dir="down", x=self.player_x, y=self.player_y, banned_positions=self.banned_positions):
                self.movable[self.player_x][self.player_y] = "."
                self.movable[self.player_x + 1][self.player_y] = "O"
                self.player_x += 1
            elif self.movable[self.player_x + 1][self.player_y] == "X" and self.can_move(dir="down", x=self.player_x + 1, y=self.player_y, banned_positions=self.non_player_positions):
                self.movable[self.player_x + 2][self.player_y] = "X"
                self.movable[self.player_x + 1][self.player_y] = "O"
                self.movable[self.player_x][self.player_y] = "."
                self.player_x += 1

        if pressed_keys[K_LEFT] and self.player_y > 0:
            if self.movable[self.player_x][self.player_y - 1] != "X" and self.can_move(dir="left", x=self.player_x, y=self.player_y, banned_positions=self.banned_positions):
                self.movable[self.player_x][self.player_y] = "."
                self.movable[self.player_x][self.player_y - 1] = "O"
                self.player_y -= 1
            elif self.movable[self.player_x][self.player_y - 1] == "X" and self.can_move(dir="left", x=self.player_x, y=self.player_y - 1, banned_positions=self.non_player_positions):
                self.movable[self.player_x][self.player_y - 2] = "X"
                self.movable[self.player_x][self.player_y - 1] = "O"
                self.movable[self.player_x][self.player_y] = "."
                self.player_y -= 1

        if pressed_keys[K_RIGHT] and self.player_y < self.columns:
            if self.movable[self.player_x][self.player_y + 1] != "X" and self.can_move(dir="right", x=self.player_x, y=self.player_y, banned_positions=self.banned_positions):
                self.movable[self.player_x][self.player_y] = "."
                self.movable[self.player_x][self.player_y + 1] = "O"
                self.player_y += 1
            elif self.movable[self.player_x][self.player_y + 1] == "X" and self.can_move(dir="right", x=self.player_x, y=self.player_y + 1, banned_positions=self.non_player_positions):
                self.movable[self.player_x][self.player_y + 2] = "X"
                self.movable[self.player_x][self.player_y + 1] = "O"
                self.movable[self.player_x][self.player_y] = "."
                self.player_y += 1


    def can_move(self, dir, x, y, banned_positions):
        if dir == "up":
            if x > 0 and (x-1 , y) not in banned_positions:
                return True
        if dir == "down":
            if x < self.rows and (x+1, y) not in banned_positions:
                return True
        if dir == "left":
            if y > 0 and (x, y-1) not in banned_positions:
                return True
        if dir == "right":
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
                    wall = pygame.image.load("/home/jedrzej/Desktop/wall.png")
                    self.surf.blit(wall, (y, x))
                    self.banned_positions.add((x // self.texture_size, y // self.texture_size))
                    y += self.texture_size
                if element == "*":
                    exit = pygame.image.load("/home/jedrzej/Desktop/exit.png")
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
                    player = pygame.image.load("/home/jedrzej/Desktop/pacman.png")
                    self.surf.blit(player, (y, x))
                    y += self.texture_size
                if element == "X":
                    crate = pygame.image.load("/home/jedrzej/Desktop/crate.png")
                    self.surf.blit(crate, (y, x))
                    self.banned_positions.add((x // self.texture_size, y // self.texture_size))
                    y += self.texture_size
            x += self.texture_size
            y = 0

def main(filepath:str, texture_size:int=64):
    world = World(filepath)
    world.load_background()
    world.load_movable()
    
    clock = pygame.time.Clock()

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
    
        pressed_keys = pygame.key.get_pressed()
        world.move_player(pressed_keys)
        world.surf.fill((255,239,213))
        clock.tick(8)
    
    font = pygame.font.Font('freesansbold.ttf', 32) 
    text = font.render("You won!", True, (139,69,19), (205,133,63)) 
    textRect = text.get_rect()  
    textRect.center = (world.y_pixels // 2, world.x_pixels // 2) 

    world.surf.blit(text, textRect)
    pygame.display.update()

pygame.init()
print(main("/home/jedrzej/Desktop/level2.txt"))