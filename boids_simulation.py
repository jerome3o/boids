from typing import List
import logging

import pygame

from controls import default_controls
from engine import CharacterEntity
from boids import BoidFlock, BoidRule, FearMePunyBoidRule, SimpleSeparationRule, AvoidWallsRule, AlignmentRule, \
    CohesionRule, NoiseRule, SpiralRule, ControlRule
from game_settings import GameSettings


logging.basicConfig()
logger = logging.getLogger(__name__)


pygame.init()


def main():
    game_settings = GameSettings()
    # game_settings.debug = True

    pygame.display.set_caption("First Game")
    win = pygame.display.set_mode((game_settings.window_width, game_settings.window_height))
    fill_colour = (0, 0, 0)

    n_boids = 50
    boid_fear = 10
    boid_radius = 100
    boid_max_speed = 100

    # character = CharacterEntity(50, 50, game_settings=game_settings, controls=default_controls, width=5,
    #                             v=200)

    flock = BoidFlock(game_settings)
    flock_rules: List[BoidRule] = [
        # FearMePunyBoidRule(weighting=1, game_settings=game_settings, entity_to_fear=character, push_force=-100),
        CohesionRule(weighting=0.5, game_settings=game_settings),
        AlignmentRule(weighting=1, game_settings=game_settings),
        NoiseRule(weighting=1, game_settings=game_settings),
        AvoidWallsRule(weighting=1, game_settings=game_settings, push_force=100),
        SimpleSeparationRule(weighting=1, game_settings=game_settings, push_force=boid_fear),
        # SpiralRule(weighting=0.01, game_settings=game_settings),
        ControlRule(weighting=1, game_settings=game_settings, control_factor=1)
    ]
    flock.generate_boids(n_boids, rules=flock_rules, local_radius=boid_radius, max_velocity=boid_max_speed,
                         controls=default_controls)

    entities = flock.boids
    tick_length = int(1000/game_settings.ticks_per_second)

    last_tick = pygame.time.get_ticks()
    while game_settings.is_running:
        win.fill(fill_colour)

        time_since_last_tick = pygame.time.get_ticks() - last_tick
        if time_since_last_tick < tick_length:
            pygame.time.delay(tick_length - time_since_last_tick)

        time_since_last_tick = pygame.time.get_ticks() - last_tick

        last_tick = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_settings.is_running = False

        keys = pygame.key.get_pressed()

        for entity in entities:
            entity.update(keys, win, time_since_last_tick/1000)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
