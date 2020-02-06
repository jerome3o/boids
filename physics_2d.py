from abc import ABC, abstractmethod

from engine import Entity


class PhysicsObject(Entity, ABC):

    @property
    @abstractmethod
    def v(self):
        pass

    @property
    @abstractmethod
    def a(self):
        pass

    @property
    @abstractmethod
    def pos(self):
        pass


