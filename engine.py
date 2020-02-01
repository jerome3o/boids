from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List

import numpy as np
import pygame

from game_settings import MapEdgeBehaviour, GameSettings


class EntityAction(Enum):
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()


class Entity(ABC):
    def __init__(self, x, y, game_settings: GameSettings, controls: Dict[int, EntityAction]=None, **kwargs):
        if controls is None:
            controls = {}

        self.x = x
        self.y = y
        self.colour = (255, 0, 0)
        self.controls = controls
        self.game_settings: GameSettings = game_settings

        self.kwargs = kwargs

    @abstractmethod
    def draw(self, win):
        pass

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @abstractmethod
    def update_physics(self, actions: List[EntityAction], time_elapsed):
        pass

    def check_physics(self):
        if self.x > self.game_settings.map_width:
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.x = self.x % self.game_settings.map_width
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.x = self.game_settings.map_width

        if self.x < 0:
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.x = self.x % self.game_settings.map_width
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.x = 0

        if self.y > self.game_settings.map_width:
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.y = self.y % self.game_settings.map_height
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.y = self.game_settings.map_height

        if self.y < 0:
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.y = self.y % self.game_settings.map_width
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.y = 0

    def parse_controls(self, keys):
        return [action for key, action in self.controls.items() if keys[key]]

    def get_debug_text(self):
        return f"pos:{self.x:0.1f}, {self.y:0.1f}"

    def draw_debug_info(self, win):
        font = pygame.font.SysFont("Courier New", 16)
        text_surface = font.render(self.get_debug_text(), True, self.game_settings.debug_text_colour)
        win.blit(text_surface, (self.x, self.y))

    def update(self, keys, win, time_elapsed):
        self.update_physics(self.parse_controls(keys), time_elapsed)
        self.check_physics()
        self.draw(win)

        if self.game_settings.debug:
            self.draw_debug_info(win)

    def distance_to(self, other):
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)


class CharacterEntity(Entity):
    def __init__(self, *args, width=5, v=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.v = v

    def draw(self, win):
        pygame.draw.circle(win, self.colour, (int(self.x), int(self.y)), self.width)

    def update_physics(self, actions: List[EntityAction], time_elapsed):
        for action in actions:
            if action == EntityAction.MOVE_UP:
                self.y -= self.v * time_elapsed
                continue
            if action == EntityAction.MOVE_DOWN:
                self.y += self.v * time_elapsed
                continue
            if action == EntityAction.MOVE_LEFT:
                self.x -= self.v * time_elapsed
                continue
            if action == EntityAction.MOVE_RIGHT:
                self.x += self.v * time_elapsed