from abc import ABC
from typing import List
import numpy as np

from engine import EntityAction
from physics_2d import PhysicsObject

import pygame


real_gravitational_constant = 6.67e-11


def calc_diff_matrix(pos):
    x_diff = pos[:, 0][np.newaxis, :] - pos[:, 0][:, np.newaxis]
    y_diff = pos[:, 1][np.newaxis, :] - pos[:, 1][:, np.newaxis]

    return np.concatenate((x_diff[:, :, np.newaxis], y_diff[:, :, np.newaxis]), axis=2)


def calc_distance_matrix(diff_matrix):
    return np.sqrt(np.sum(diff_matrix ** 2, axis=2))


def calc_direction_matrix(diff_matrix, distance_matrix):
    return diff_matrix/distance_matrix[:, :, np.newaxis]


def calc_mass_product_matrix(m):
    return m[:, 0][np.newaxis, :] - m[:, 0][:, np.newaxis]


class GravitationalField:
    def __init__(self, g=real_gravitational_constant):
        self.g = real_gravitational_constant
        self._gravitational_objects = []
        self._v = np.ndarray([0, 2])
        self._pos = np.ndarray([0, 2])
        self._f = np.ndarray([0, 2])
        self._m = np.ndarray([0, 1])

    @property
    def v(self):
        return self._v

    @property
    def pos(self):
        return self._pos

    @property
    def f(self):
        return self._f

    @property
    def m(self):
        return self._m

    def add_gravitational_object(self, gravitational_object, pos, v, f, m):
        if gravitational_object in self._gravitational_objects:
            return self._gravitational_objects.index(gravitational_object)

        self._gravitational_objects.append(gravitational_object)
        self._f = np.concatenate((self._f, f[np.newaxis, :]), axis=0)
        self._v = np.concatenate((self._v, v[np.newaxis, :]), axis=0)
        self._pos = np.concatenate((self._pos, pos[np.newaxis, :]), axis=0)
        self._m = np.concatenate((self._m, m[np.newaxis, :]), axis=0)

        return len(self._gravitational_objects) - 1

    def update(self, time_passed):
        diff_matrix = calc_diff_matrix(self._pos)
        distance_matrix = calc_distance_matrix(diff_matrix)
        direction_matrix = calc_direction_matrix(diff_matrix, distance_matrix)
        mass_product_matrix = calc_mass_product_matrix(self._m)

        g_force_magnitude_matrix = self.g * mass_product_matrix / distance_matrix**2
        g_force_matrix = g_force_magnitude_matrix[:, :, np.newaxis] * direction_matrix

        f = np.sum(g_force_matrix, axis=1)
        self._f = f
        self._v += (self.f / self.m) * time_passed
        self._pos = self._v * time_passed


class GravitationalObject(PhysicsObject, ABC):
    def __init__(self, *args,  gravitational_field: GravitationalField,
                 pos=np.array([0, 0]), v=np.array([0, 0]), f=np.array([0, 0]),
                 m=np.array([1000]), **kwargs):
        super().__init__(*args, **kwargs)
        self.gravitational_field = gravitational_field
        self.g_index = gravitational_field.add_gravitational_object(self, pos, v, f, m)

    @property
    def v(self):
        return self.gravitational_field.v[self.g_index, :]

    @property
    def pos(self):
        return self.gravitational_field.pos[self.g_index, :]

    @property
    def m(self):
        return self.gravitational_field.m[self.g_index, :]

    @property
    def f(self):
        return self.gravitational_field.f[self.g_index, :]

    @property
    def a(self):
        return self.f / self.m

    def update_physics(self, actions: List[EntityAction], time_elapsed):
        pass


class Planet(GravitationalObject):
    def __init__(self, *args, size=100, **kwargs):
        self.size = size
        super().__init__(*args, **kwargs)

    def draw(self, win):
        pygame.draw.circle(win, self.colour, self.pos, self.size)
        pygame.draw.line(win, (255, 255, 0), self.pos, self.pos + self.v)
        pygame.draw.line(win, (255, 0, 0), self.pos, self.pos + self.a)

