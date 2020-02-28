import random
from abc import ABC, abstractmethod
from typing import List
import logging

import numpy as np
import pygame

from engine import Entity, EntityAction
from physics_2d import PhysicsObject

from game_settings import GameSettings


logger = logging.getLogger(__name__)


class BoidFlock:
    def __init__(self, game_settings: GameSettings):
        self._boids: List[Boid] = None
        self.game_settings = game_settings

    def generate_boids(self, n_boids, rules=None, **kwargs):
        self._boids = [
            Boid(
                pos=np.array([random.randint(0, self.game_settings.map_width),
                              random.randint(0, self.game_settings.map_height)]),
                game_settings=self.game_settings,
                rules=rules,
                flock=self,
                **kwargs,
            )
            for _ in range(n_boids)
        ]

    @property
    def boids(self):
        return self._boids

    def get_local_boids(self, boid: Entity):
        return [other_boid for other_boid in self.boids
                if boid.distance_to(other_boid) < boid.local_radius and boid != other_boid]


class Boid(PhysicsObject):

    @property
    def a(self):
        return 0

    @property
    def pos(self):
        return self._pos

    def __init__(self, *args, flock: BoidFlock, colour=None, rules=None, size=10, local_radius=200, max_velocity=30,
                 speed=20, **kwargs):
        super().__init__(*args, **kwargs)

        if colour is None:
            colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if rules is None:
            rules = list()

        self.colour = colour
        self.flock = flock
        self.size = size

        self.local_radius = local_radius
        self.max_velocity = max_velocity
        self.speed = speed
        self._v = np.array([0, 0])

        self.rules = rules
        self.n_neighbours = 0

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):

        magnitude = np.linalg.norm(v)
        if magnitude > self.max_velocity:
            v = v * (self.max_velocity/magnitude)

        self._v = v

    def draw(self, win):
        if abs(self.v).sum() == 0:
            direction = np.array([0, 1])
        else:
            direction = self.v / np.linalg.norm(self.v)

        direction *= self.size
        perpendicular_direction = np.cross(np.array([*direction, 0]), np.array([0, 0, 1]))[:2]

        centre = self.pos

        points = [
            0.5*direction + centre,
            -0.5*direction + 0.25*perpendicular_direction + centre,
            -0.25*direction + centre,
            -0.5*direction - 0.25*perpendicular_direction + centre,
        ]

        pygame.draw.polygon(win, self.colour, points)

    def update_physics(self, actions: List[EntityAction], time_elapsed):

        local_boids: List[Boid] = self.flock.get_local_boids(self)

        direction = self.calculate_rules(local_boids, actions)
        self.n_neighbours = len(local_boids)

        self.v = self.v + direction * self.speed

        self._pos += (self.v * time_elapsed).astype(int)

        # TODO: Game clock

    def get_debug_text(self):
        return super().get_debug_text() + f", n={self.n_neighbours}"

    def calculate_rules(self, local_boids, actions):
        return sum(
            [rule.evaluate(self, local_boids, actions=actions) * rule.weight for rule in self.rules]
        )


class BoidRule(ABC):
    _name = "BoidRule"

    def __init__(self, weighting: float, game_settings: GameSettings):
        self._weight = weighting
        self.game_settings = game_settings

    @abstractmethod
    def _evaluate(self, boid: Boid, local_boids: List[Boid], actions: List[EntityAction]):
        pass

    def evaluate(self, boid, local_boids: List[Boid], actions):
        output = self._evaluate(boid, local_boids, actions=actions)
        if np.isnan(output).any():
            logger.warning(f"NaN encountered in {self.name}")
            return np.array([0, 0])
        return output

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value


class SimpleSeparationRule(BoidRule):
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force

    _name = "Separation"

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        n = len(local_boids)
        if n > 1:
            direction_offsets = np.array([(boid.pos - other_boid.pos) for other_boid in local_boids])
            magnitudes = np.sum(np.abs(direction_offsets)**2, axis=-1)**(1./2)
            normed_directions = direction_offsets / magnitudes[:, np.newaxis]
            v = np.sum(normed_directions * (self.push_force/magnitudes)[:, np.newaxis], axis=0)
        else:
            v = np.array([0, 0])

        return v


class AlignmentRule(BoidRule):
    _name = "Alignment"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Boid, local_boids, **kwargs):
        other_velocities = np.array([b.v for b in local_boids])

        if len(other_velocities) == 0:
            return np.array([0, 0])

        magnitudes = np.sum(np.abs(other_velocities) ** 2, axis=-1) ** (1. / 2)
        normed_directions: np.ndarray = other_velocities / magnitudes[:, np.newaxis]

        v: np.ndarray = normed_directions.mean(axis=0)
        return v


class CohesionRule(BoidRule):
    _name = "Cohesion"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        if len(local_boids) == 0:
            return np.array([0, 0])
        average_pos = np.array([b.pos for b in local_boids]).mean(axis=0)
        diff = average_pos - boid.pos
        mag = np.sqrt((diff**2).sum())
        if mag == 0:
            return np.array([0, 0])
        return diff / mag


class AvoidWallsRule(BoidRule):
    def __init__(self, *args, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force

    def _evaluate(self, boid: Boid, local_boids, **kwargs):
        fake_boids = np.array([
            [0, boid.pos[1]],
            [self.game_settings.map_width, boid.pos[1]],
            [boid.pos[0], 0],
            [boid.pos[0], self.game_settings.map_height],
        ])

        direction_offsets = boid.pos - fake_boids
        magnitudes = np.sum(np.abs(direction_offsets) ** 2, axis=-1) ** (1. / 2)
        normed_directions = direction_offsets / magnitudes[:, np.newaxis]
        adjusted_magnitudes = magnitudes**2
        v = np.sum(normed_directions * (self.push_force / adjusted_magnitudes)[:, np.newaxis], axis=0)

        return v


class MoveRightRule(BoidRule):
    _name = "MoveRight"

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        return np.array([10, 0])


class FearMePunyBoidRule(BoidRule):
    _name = "FearMePunyBoid"

    def __init__(self, *args, entity_to_fear, push_force=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_force = push_force
        self.fearful_entity = entity_to_fear

    def _evaluate(self, boid: Boid, local_boids, **kwargs):
        direction_offsets = boid.pos - self.fearful_entity.pos
        magnitude = np.sum(np.abs(direction_offsets) ** 2, axis=-1) ** (1. / 2)
        normed_directions = direction_offsets / magnitude

        adjusted_magnitude = (self.push_force / magnitude) ** 2
        if self.push_force < 0:
            adjusted_magnitude *= -1.0

        return normed_directions * adjusted_magnitude


class NoiseRule(BoidRule):
    _name = "Noise"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid, local_boids: List[Boid], **kwargs):
        return np.random.uniform(-1, 1, 2)


class SpiralRule(BoidRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Boid, local_boids: List[Boid], **kwargs):
        return np.cross(boid.v, np.array([0, 0, 1]))[:2]


class ControlRule(BoidRule):
    def __init__(self, *args, control_factor, **kwargs):
        self.control_factor = control_factor
        super().__init__(*args, **kwargs)

    def _evaluate(self, boid: Boid, local_boids: List[Boid], actions: List[EntityAction]=None):
        v = np.array([0, 0])

        if actions is None:
            return v

        for action in actions:
            if action == EntityAction.MOVE_UP:
                v[1] -= self.control_factor
                continue
            if action == EntityAction.MOVE_DOWN:
                v[1] += self.control_factor
                continue
            if action == EntityAction.MOVE_LEFT:
                v[0] -= self.control_factor
                continue
            if action == EntityAction.MOVE_RIGHT:
                v[0] += self.control_factor

        return v
