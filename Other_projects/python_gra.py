import pygame
from pygame.locals import *


pygame.init()


class World(pygame.Surface):

    def __init__(self, filename:str):
        with open(filename, "r") as handle:
            level = handle.read()
        self.level = level
        lines = [line for line in level.split("\n") if line != ""]
        self.rows = len(lines)
        self.columns = len(lines[0])
        self.x = self.rows * 64
        self.y = self.columns * 64
        self.surf = pygame.display.set_mode((self.y, self.x))
        self.elements = []
        self.load_level()
        self.surf.fill((255,255,255))
        
        
    def load_level(self):
        lines = [line for line in self.level.split("\n") if line != ""]
        x = 0
        y = 0
        for line in lines:
            for field in line:
                if field == "#":
                    wall = self.make_wall(x, y)
                    self.elements.append(wall)
                    x += 64

                if field == ".":
                    x +=64

                if field == "X":
                    crate = self.make_crate(x, y)
                    self.elements.append(crate)
                    x += 64
                
                if field == "O":
                    player = self.make_player(x, y)
                    self.player = player
                    x += 64

                if field == "*":
                    exit = self.make_exit(x, y)
                    self.elements.append(exit)
                    x += 64
            x = 0
            y += 64

    def make_wall(self, x, y):
        wall = Wall()
        wall.x = x
        wall.y = y
        return wall
    
    def make_crate(self, x, y):
        crate = Crate()
        crate.x = x
        crate.y = y
        return crate
    
    def make_player(self, x, y):
        player = Player()
        player.x = x
        player.y = y
        return player
                
    def make_exit(self, x, y):
        exit = Exit()
        exit.x = x
        exit.y = y
        return exit

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((64, 64))
        self.surf.fill((210, 100, 115))
        self.rect = self.surf.get_rect()
    
        
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -64)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 64)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-64, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(64, 0)
    
class Wall(pygame.sprite.Sprite):
    def __init__(self):
        super(Wall, self).__init__()
        self.surf = pygame.Surface((64, 64))
        self.surf.fill((50, 50, 50))
        self.rect = self.surf.get_rect()

class Crate(pygame.sprite.Sprite):
    def __init__(self):
        super(Crate, self).__init__()
        self.surf = pygame.Surface((64, 64))
        self.surf.fill((125, 15, 90))
        self.rect = self.surf.get_rect()

class Exit(pygame.sprite.Sprite):
    def __init__(self):
        super(Exit, self).__init__()
        self.surf = pygame.Surface((64, 64))
        self.surf.fill((75, 35, 180))
        self.rect = self.surf.get_rect()


def main():
    world = World("/home/jedrzej/Desktop/level.txt")
    running = True
    

    while running is True:

        world.surf.fill((255,255,255))
        for element in world.elements:
            world.surf.blit(element.surf, (element.x, element.y))

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_UP:
                    world.surf.fill(( 30, 30, 30))
    
        pressed_keys = pygame.key.get_pressed()
        
        world.player.update(pressed_keys)
        world.surf.blit(world.player.surf, world.player.rect)
        # print("FLIPPITY FLOPPITY")
        pygame.display.update()
        clock = pygame.time.Clock()
        clock.tick(15)
        # print(world.player.rect)
main()
