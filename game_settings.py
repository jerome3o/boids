from enum import Enum, auto


class MapEdgeBehaviour(Enum):
    WRAP = auto()
    FREE = auto()
    CLAMP = auto()


class GameSettings:
    def __init__(self):
        self.window_width = 1800
        self.window_height = 800

        self.map_width = 1800
        self.map_height = 800

        self.is_running = True
        self.ticks_per_second = 144

        self.x_edge_behaviour = MapEdgeBehaviour.WRAP
        self.y_edge_behaviour = MapEdgeBehaviour.WRAP

        self.debug = False
        self.debug_text_colour = (255, 255, 255)