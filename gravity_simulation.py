from typing import List

import pygame
import numpy as np

from controls import default_controls
from game_settings import GameSettings
from gravity import GravitationalObject, GravitationalField, Planet, MySpaceship

import matplotlib.pyplot as plt

rng = np.random.randint


def lerp(x, lower, upper):
    return (1 - x) * lower + x * upper


def inv_lerp(value, upper, lower):
    return (value - lower) / (upper - lower)


def get_random_planet(game_settings, g_field,
                      size_range=(40, 200),
                      pos_range=((-2000, 2000), (-2000, 2000)),
                      v_range=((-100, 100), (-100, 100)),
                      density_range=(1000000, 10000000),
                      cmap=plt.get_cmap("jet")):

    size = rng(*size_range)
    pos = np.array([rng(pos_range[0][0], pos_range[0][1]), rng(pos_range[1][0], pos_range[1][1])])
    v = np.array([rng(v_range[0][0], v_range[0][1]), rng(v_range[1][0], v_range[1][1])])
    density = rng(density_range[0], density_range[1])

    m = np.array([density * size])

    density_percent = inv_lerp(density, *density_range)
    colour = [int(x * 255) for x in cmap(int(density_percent*255))][:3]

    return Planet(game_settings,
                  gravitational_field=g_field,
                  size=size,
                  pos=pos,
                  v=v,
                  controls={},
                  m=m,
                  colour=colour)


def main():

    win_size = 1000
    game_settings = GameSettings()
    game_settings.window_height = win_size
    game_settings.window_width = win_size
    game_settings.map_height = win_size
    game_settings.map_width = win_size
    game_settings.zoom = 0.5

    game_settings.zoom_factor = 0.005

    pygame.init()
    win: pygame.Surface = pygame.display.set_mode((game_settings.window_width, game_settings.window_height))

    g_field = GravitationalField(g=0.001)

    n_planets = 50

    g_objects: List[GravitationalObject] = [
        get_random_planet(game_settings, g_field,
                          v_range=((0, 1), (0, 1)),
                          pos_range=((-10000, 10000), (-10000, 10000)),
                          density_range=(10000000, 100000000),
                          size_range=(50, 200))
        for _ in range(n_planets)
    ]
    g_objects.append(MySpaceship(game_settings, gravitational_field=g_field, size=10,
                                 m=np.array([10]),
                                 colour=(0, 0, 100),
                                 controls=default_controls))

    tick_time = 1/game_settings.ticks_per_second * 1000

    while game_settings.is_running:
        win.fill((0, 0, 0))
        pygame.time.delay(int(tick_time))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_settings.is_running = False

        keys = pygame.key.get_pressed()

        font = pygame.font.SysFont("Courier New", 16)
        text_surface = font.render(f"camera pos: ({game_settings.camera_pos[0]: 0.2f},{game_settings.camera_pos[1]: 0.2f})\r"
                                   f"zoom: {game_settings.zoom: 0.2f}", True, game_settings.debug_text_colour)
        win.blit(text_surface, (10, 10))

        if keys[pygame.K_q]:
            game_settings.zoom *= 1 - game_settings.zoom_factor
        if keys[pygame.K_e]:
            game_settings.zoom *= 1 + game_settings.zoom_factor

        if keys[pygame.K_w]:
            game_settings.camera_pos[1] += 10
        if keys[pygame.K_s]:
            game_settings.camera_pos[1] -= 10
        if keys[pygame.K_a]:
            game_settings.camera_pos[0] += 10
        if keys[pygame.K_d]:
            game_settings.camera_pos[0] -= 10

        for obj in g_objects:
            obj.update_physics(obj.parse_controls(keys), time_elapsed=tick_time)

            if type(obj) == MySpaceship:
                obj.centre()

            obj.draw(win)

        g_field.update(tick_time/1000)
        pygame.display.flip()


if __name__ == '__main__':
    main()
