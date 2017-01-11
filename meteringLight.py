import pyglet
import numpy as np


class MeterLight(object):
    length_px = 20
    c_red = (247, 70, 29)
    c_green = (42, 203, 13)

    def __init__(self, dist2ramp_start, red, ramp_start, scale):

        self.p = ramp_start - dist2ramp_start
        self.comm_range = 430 - self.p  # to the ramp end
        self.scale = scale

        # controlled by light
        self.t = 0
        self.min_red_t = red
        self.on_green = True
        self.main_av_speed1 = 90/3.6
        self.main_av_speed2 = 10/3.6
        self.main_av_redu2speed = self.main_av_speed1
        self.main_av_decel = -2

        # controlled by av
        self.m_sensitive_speed = 70/3.6
        self.r_sensitive_speed = 10/3.6

        y = 70
        x = self.scaling(450-self.p, 'm')

        self.vertex_list = pyglet.graphics.vertex_list(4,
                      ('v2i', (x, y,
                               x+self.length_px, y,
                               x+self.length_px, y-self.length_px,
                               x, y-self.length_px)),
                      ('c3B', self.c_green*4)
                      )
        self.vertex_list.draw(pyglet.gl.GL_QUADS)

    def update(self, r_active, m_active, dt, AV_RATE):
        if AV_RATE == 0:
            self.on_green = True
        else:
            # av control light
            m_in_range = m_active[(m_active['p']>self.p-self.comm_range) & (m_active['p']<self.p+self.comm_range)]
            m_av_v_mean = m_in_range.loc[m_active['AV'], 'v'].mean()
            r_in_upper_range = r_active[(r_active['p']>self.p+10) & (r_active['p']<self.p+self.comm_range)]
            r_av_v_mean = r_in_upper_range.loc[r_active['AV'], 'v'].mean()
            if np.isnan(m_av_v_mean): m_av_v_mean = 200
            if np.isnan(r_av_v_mean): r_av_v_mean = 200

            n_r_after_meter = r_active[r_active['p']>self.p].shape[0]
            if (m_av_v_mean > self.m_sensitive_speed) and (r_av_v_mean > self.r_sensitive_speed) and (n_r_after_meter<2):
                # requirement :
                # 1. main_av average speed > m sensitive speed
                # 2. ramp av average speed > r sensitive speed
                # 3. let maximum 2 ramp waiting car pass as one time
                self.on_green = True
                self.vertex_list.colors = self.c_green * 4
            else:
                self.on_green = False
                self.vertex_list.colors = self.c_red * 4

        # if r_active[r_active['p']>self.p].shape[0]<=1:
        #     mean_v = r_active.loc[r_active['p']>self.p, 'v'].mean()
        #     if mean_v > 10/3.6 or np.isnan(mean_v):
        #         if self.t > self.min_red_t:
        #             self.t -= self.min_red_t
        #             # no more than 2 cars merging, merging cars' speed should > 10 km/h
        #             self.on_green = True
        #             self.vertex_list.colors = self.c_green * 4
        #             self.main_av_redu2speed = self.main_av_speed1
        #         else:
        #             self.t += dt
        # else:
        #     self.on_green = False
        #     self.vertex_list.colors = self.c_red * 4
        #     self.main_av_redu2speed = self.main_av_speed2


    def scaling(self, value, unit):
        # scale to pixel: 1m:5px
        if unit in ["m/s", 'm']:
            v = value * self.scale
        elif unit == "km/h":
            v = value / (3.6 / self.scale)
        return v