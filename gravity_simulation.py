from typing import List

import pygame

from game_settings import GameSettings
from gravity import GravitationalObject, GravitationalField, Planet


def main():

    game_settings = GameSettings()

    pygame.init()
    win = pygame.display.set_mode((game_settings.window_width, game_settings.window_height))

    g_field = GravitationalField()

    g_objects: List[GravitationalObject] = [
        Planet()
    ]



if __name__ == '__main__':
    main()
