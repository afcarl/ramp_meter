import numpy as np
import pyglet
import matplotlib.pyplot as plt

AV_IMG = pyglet.image.load('av.png')


class Car(object):
    length = 4
    width = 2
    radius = np.sqrt(np.square(length / 2) + np.square(width / 2))

    def __init__(self, id, p, v, is_av, scale, road_sections, batch):
        self.id = id
        self.scale = scale
        self.anchor_x = 0
        self.anchor_y = self.scaling(road_sections[1], 'm')/np.pi*2+100
        self.road_sections = np.cumsum(np.array(road_sections))
        self.batch = batch
        self.rotation = 0
        self.p = p
        self.v = v
        self.is_av = is_av

        self.vertex_list = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f/stream', self.vertices_coord),
            ('c3f/stream', (0, 0, 255,)*4)
        )
        if is_av:
            c1 = self.scaling(1,'m')
            self.av_sprite = pyglet.sprite.Sprite(AV_IMG, x=self.anchor_x-c1, y=self.anchor_y, batch=self.batch)
            self.av_sprite.scale = 0.05

    @property
    def vertices_coord(self):
        beta = np.arctan(self.width / self.length)
        gamma1 = beta+self.rotation
        gamma2 = beta-self.rotation
        dx1 = self.radius_px*np.cos(gamma1)
        dy1 = self.radius_px*np.sin(gamma1)
        dx2 = self.radius_px*np.cos(gamma2)
        dy2 = self.radius_px*np.sin(gamma2)
        x0, y0 = self.anchor_x - dx2, self.anchor_y + dy2
        x1, y1 = self.anchor_x + dx1, self.anchor_y + dy1
        x2, y2 = self.anchor_x + dx2, self.anchor_y - dy2
        x3, y3 = self.anchor_x - dx1, self.anchor_y - dy1
        return (x0, y0, x1, y1, x2, y2, x3, y3)

    def update_av(self):
        if self.is_av:
            c1 = self.scaling(1, 'm')
            self.av_sprite.set_position(self.anchor_x-c1, self.anchor_y)

    @property
    def width_px(self):
        return self.scaling(self.width, 'm')

    @property
    def length_px(self):
        return self.scaling(self.length, 'm')

    @property
    def radius_px(self):
        return self.scaling(self.radius, 'm')

    def scaling(self, value, unit):
        # scale to pixel: 1m:5px
        if unit in ["m/s", 'm']:
            v = value * self.scale
        elif unit == "km/h":
            v = value / (3.6 / self.scale)
        return v

    def get_rgb(self):
        return plt.cm.jet(1 - self.v / 34)[:-1] * 4


class RCar(Car):
    def __init__(self, id, p, v, is_av, scale, road_sections, batch):
        super(RCar, self).__init__(id, p, v, is_av, scale, road_sections, batch)
        self.merged = False

    def update(self):
        shift = 100 if self.merged else 100-self.scaling(3, 'm')

        self.anchor_x = self.scaling(self.road_sections[2] - self.p, 'm')
        self.anchor_y = shift

        self.vertex_list.vertices = self.vertices_coord
        self.update_av()

        c_values = self.get_rgb()
        self.vertex_list.colors = c_values


class MCar(Car):

    def __init__(self, id, p, v, is_av, scale, road_sections, batch):
        super(MCar, self).__init__(id, p, v, is_av, scale, road_sections, batch)

    def update(self):
        if self.p < self.road_sections[0]:
            self.anchor_x = self.scaling(self.p, 'm')
        elif self.p > self.road_sections[1]:
            self.anchor_x = self.scaling(self.road_sections[2] - self.p, 'm')
            self.anchor_y = 100
            self.rotation = np.pi
        else:  # in the circle
            radius = self.scaling(self.road_sections[1] - self.road_sections[0], 'm')/np.pi
            rad = self.scaling(self.p - self.road_sections[0], 'm')/radius
            self.rotation = -rad
            self.anchor_x = radius * np.sin(rad) + self.scaling(self.road_sections[0], 'm')
            self.anchor_y = radius * np.cos(rad) + 100 + radius

        self.vertex_list.vertices = self.vertices_coord
        self.update_av()

        c_values = self.get_rgb()
        self.vertex_list.colors = c_values
