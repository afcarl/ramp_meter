import pyglet
import numpy as np
import pandas as pd
from car import MCar, RCar
from meteringLight import MeterLight


SCALE = 5


def scaling(value, unit):
    # scale to pixel: 1m:5px
    if unit in ["m/s", 'm']:
        v = value*SCALE
    elif unit == "km/h":
        v = value/(3.6 / SCALE)
    return v

def red_time():
    return 3600/RAMP_VOLUME


class Window(pyglet.window.Window):
    def __init__(self, width=1100, height=600):
        super().__init__(width, height, resizable=True, caption="Metering Simulation", vsync=False)
        pyglet.clock.schedule_interval(self.update, 1 / 60.)

        self.batch_labels = pyglet.graphics.Batch()
        label_color = (62, 147, 184, 255)
        self.label_speed = pyglet.text.Label('Running speed (Q, A): %.1fx' % float(SPEED_UP), font_size=12,
                          x=20, y=400, color=label_color, batch=self.batch_labels)
        self.label_main_volume = pyglet.text.Label('Main road (W, S): %i (veh/h)' % MAIN_VOLUME, font_size=12,
                                             x=20, y=380, color=label_color, batch=self.batch_labels)
        self.label_ramp_volume = pyglet.text.Label('Ramp (E, D): %i (veh/h)' % RAMP_VOLUME, font_size=12,
                                                   x=20, y=360, color=label_color, batch=self.batch_labels)
        # self.label_light = pyglet.text.Label('Red light: %.1f (s)' % RED, font_size=12,
        #                                            x=20, y=340, color=label_color, batch=self.batch_labels)
        self.label_av_rate = pyglet.text.Label('AV rate (R, F): %.2f ' % AV_RATE, font_size=12,
                                             x=20, y=320, color=label_color, batch=self.batch_labels)
        self.label_merged_volume = pyglet.text.Label('Merged volume: %.1f (veh/h)' % MAIN_VOLUME, font_size=12,
                                               x=20, y=300, color=label_color, batch=self.batch_labels)

        self.sec1 = section1 = 150
        self.sec2 = section2 = 150
        self.ramp_start = 370
        self.ramp_end = 430

        # this road length 450=150+150+150
        lc = [[0, 100], [scaling(section1, 'm'), 100],
              [scaling(section1, 'm'), scaling(section2, 'm')/np.pi*2+100], [0, scaling(section2, 'm')/np.pi*2+100]]
        self.build_road(lc)
        self.fps = pyglet.clock.ClockDisplay(color=(0,1,1,0.5))

        # meter config
        self.meter_light = MeterLight(
            dist2ramp_start=50, ramp_start=self.ramp_start,
            red=RED, scale=SCALE)

        # dims = (time, car_properties)
        self.main_active = pd.DataFrame(columns=['p', 'v', 'a', 'ramp', 'AV', 'signaled'])
        self.ramp_active = self.main_active.copy()
        self.tracking_p = pd.DataFrame([])  # (time, car_id)
        self.acc_inter_main = 0
        self.acc_inter_ramp = 0
        self.merged_volume = MAIN_VOLUME
        self.t = 0

        # 1st main road car
        self.batch_car = pyglet.graphics.Batch()
        c_id = self.get_new_global_id()
        self.m_cars = [
            MCar(id=c_id, p=0, v=100/3.6, is_av=False, scale=SCALE,
                road_sections=[self.sec1, self.sec2, self.sec1], batch=self.batch_car)]

        car_init = pd.Series({"p": 0, 'v': 100/3.6, 'a': 0, 'ramp': False, 'AV': False, 'signaled': False}, name=c_id)
        self.main_active = self.main_active.append(car_init)

        # 1st ramp car
        c_id = self.get_new_global_id()
        self.r_cars = [
            RCar(id=c_id, p=section1+section2-80, v=100/3.6, is_av=False, scale=SCALE,
                road_sections=[self.sec1, self.sec2, self.sec1], batch=self.batch_car)]

        car_init = pd.Series({"p": section1+section2-80, 'v': 100 / 3.6, 'a': 0, 'ramp': True, 'AV': False, 'signaled': False}, name=c_id)
        self.ramp_active = self.ramp_active.append(car_init)

    def get_new_global_id(self):
        if not hasattr(self, 'new_car_global_id'):
            self.new_car_global_id = 0
        else:
            self.new_car_global_id += 1
        return self.new_car_global_id

    def generate_car_main(self, dt):
        if not hasattr(self, 'gen_inter_main'):
            # main road volume = 1600 ~ 2000
            mean_inter = 3600/MAIN_VOLUME
            # mean_speed = DESIRED_SPEED/3.6
            last_main_car_speed = self.main_active.iloc[-3:, 1].mean()
            self.gen_inter_main = np.clip(np.random.normal(mean_inter, 1), 0.5, mean_inter*2)   # headway
            self.gen_v_main = np.clip(np.random.normal(last_main_car_speed, 1), 10/3.6, 120/3.6)   # speed
            # self.gen_inter_main = mean_inter
            # self.gen_v_main = mean_speed

        self.acc_inter_main += dt
        if (self.acc_inter_main >= self.gen_inter_main) and (self.main_active['p'].iloc[-1] > 0):
            self.acc_inter_main = 0
            c_id = self.get_new_global_id()
            is_av = True if np.random.uniform() < AV_RATE else False
            self.m_cars.append(
                MCar(id=c_id, p=-20, v=self.gen_v_main, is_av=is_av, scale=SCALE,
                    road_sections=[self.sec1, self.sec2, self.sec1], batch=self.batch_car)
            )
            car_init = pd.Series({"p": -20, 'v': self.gen_v_main, 'a': 0, 'ramp': False,
                                  'AV': is_av, 'signaled': False}, name=c_id)
            self.main_active = self.main_active.append(car_init)

            del self.gen_inter_main, self.gen_v_main  # delete for next generation

    def generate_car_ramp(self, dt):

        if not hasattr(self, 'gen_inter_ramp'):
            # ramp road volume = 400 ~ 500
            mean_inter = 3600/RAMP_VOLUME
            mean_speed = DESIRED_SPEED/3.6
            self.gen_inter_ramp = np.clip(np.random.normal(mean_inter, 2), 0.3, mean_inter*2)   # headway
            self.gen_v_ramp = np.clip(np.random.normal(mean_speed, 1), 60/3.6, 120/3.6)   # speed
            # self.gen_inter_ramp = mean_inter
            # self.gen_v_ramp = mean_speed

        def gen():
            self.acc_inter_ramp -= self.gen_inter_ramp
            c_id = self.get_new_global_id()
            is_av = True if np.random.uniform() < AV_RATE else False
            self.r_cars.append(
                RCar(id=c_id, p=self.sec1 + self.sec2 - 80, v=self.gen_v_ramp,
                     is_av=is_av, scale=SCALE,
                     road_sections=[self.sec1, self.sec2, self.sec1], batch=self.batch_car)
            )
            car_init = pd.Series({"p": self.sec1 + self.sec2 - 80, 'v': self.gen_v_ramp,
                                  'a': 0, 'ramp': True, 'AV': is_av, 'signaled': False},
                                 name=c_id)
            self.ramp_active = self.ramp_active.append(car_init)
            del self.gen_inter_ramp, self.gen_v_ramp  # delete for next generation

        self.acc_inter_ramp += dt
        if self.acc_inter_ramp >= self.gen_inter_ramp:
            if len(self.ramp_active) > 0:  # start to generate ramp car
                if self.ramp_active['p'].iloc[-1] > (self.sec1+self.sec2-70):
                    gen()
            else:
                gen()

    def update(self, dt):
        dt *= SPEED_UP
        if dt > 0.1: dt = 0.1
        self.t += dt
        self.generate_car_main(dt)  # generating cars
        self.generate_car_ramp(dt)
        self.meter_light.update(self.ramp_active, self.main_active, dt, AV_RATE) # change meter light color

        if self.main_active.shape[0] > 1:   # main road car update
            self.IDM(self.main_active, dt, is_ramp=False)
            if self.main_active['p'].iloc[0] > (self.sec1*2+self.sec2):  # pop leader
                self.main_active = self.main_active.iloc[1:, :]
                self.m_cars[0].vertex_list.delete()
                if self.m_cars[0].is_av: self.m_cars[0].av_sprite.delete()
                del self.m_cars[0]

                if not hasattr(self, 'pop_t'):
                    self.pop_t = self.t
                else:
                    # moving average merged volume
                    self.merged_volume = 0.05*(3600 / (self.t-self.pop_t)) + 0.95 * self.merged_volume
                    self.pop_t = self.t
                    self.label_merged_volume.text = 'Merged volume: %.1f (veh/h)' % self.merged_volume
        else:
            self.main_active['p'] += self.main_active['v'] * dt

        if self.ramp_active.shape[0] > 0:
            self.IDM(self.ramp_active, dt, is_ramp=True)  # ramp car update drive
            self.lane_change()  # check merging

        # graphic update
        for car in self.m_cars:
            car.p = self.main_active.loc[car.id, 'p']
            car.v = self.main_active.loc[car.id, 'v']
            car.update()

        for car in self.r_cars:
            car.p = self.ramp_active.loc[car.id, 'p']
            car.v = self.ramp_active.loc[car.id, 'v']
            car.update()

    def IDM(self, data, dt, is_ramp):
        a = 1.2 if not is_ramp else 3
        b = 2
        s0 = 2
        T = 1
        v0 = 100 / 3.6
        c = 0.99  # CAH
        lambda_dx = 1  # ramp veh

        # kick off the leader
        if not is_ramp:
            dv = (data['v'].diff()).iloc[1:]
            dx = (-data['p'].diff() - 4).iloc[1:]   # clear gap
            vn = data['v'].iloc[1:]

            s_star = s0 + np.maximum(0, vn*T + vn*dv / (2 * np.sqrt(a*b)))
            s_star_div_dx = np.append([0], s_star/dx)     # leader's s_star = 0

            vl_al = data[['v', 'a']].copy()
            vl_al = vl_al.shift(1)
            vl_al.iloc[0, :] = [100, 100]
            al, vl = vl_al['a'], vl_al['v']
            al_ = np.minimum(data['a'], al)
            a_cah = data['a']*0
            dx_ = np.append([100], dx)
            vn_ = data['v']
            a_cah[vl * (vn_ - vl) <= -2 * dx_ * al_] = np.square(vn_)*al_/(np.square(vl)-2*dx_*al_)
            a_cah[vl * (vn_ - vl) > -2 * dx_ * al_] = al_ - np.square(vn_-vl)*np.maximum(0, vn_-vl)/(2*dx_)

        else:
            dv = (data['v'].diff()).fillna(data['v'].iloc[0])
            dx = (-data['p'].diff() - 4).fillna(self.ramp_end - data['p'].iloc[0] - 2)  # clear gap
            dx *= lambda_dx
            vn = data['v']
            if (not self.meter_light.on_green) and (data['p'] < self.meter_light.p-5).any():
                index = dv[data['p'] < self.meter_light.p].index[0]
                dv[index] = data['v'][index]
                dx[index] = self.meter_light.p - data['p'][index]

            s_star = s0 + np.maximum(0, vn * T + vn * dv / (2 * np.sqrt(a * b)))
            s_star_div_dx = s_star / dx

        a_idm = a * (1 - np.power(data['v'] / v0, 4) - np.square(s_star_div_dx))

        if not is_ramp: # update CAH
            a_idm_ = a_idm[a_idm<a_cah]
            a_cah_ = a_cah[a_idm<a_cah]
            a_idm[a_idm < a_cah] = (1-c)*a_idm_ + c*(a_cah_+b*np.tanh((a_idm_-a_cah_)/b))

            # # meter communicates
            # if self.meter_light.on_green and (AV_RATE > 0):
            #     dist2meter = self.meter_light.p - data['p']  # communicative range
            #     # in communication range
            #     data.loc[(dist2meter < self.meter_light.comm_range) & (dist2meter > -self.meter_light.comm_range)
            #              & data['AV'], 'signaled'] = True  # signal
            # # meter communicates
            # if AV_RATE > 0:
            #     # decelerate in the communication range
            #     data.loc[(data['p']>self.meter_light.p+self.meter_light.comm_range), 'signaled'] = False
            #     satisfied = data['signaled'] & (data['p']<self.meter_light.p+self.meter_light.comm_range)
            #     a_idm[(data['v'] >= self.meter_light.main_av_redu2speed)
            #           & satisfied] = np.minimum(a_idm, self.meter_light.main_av_decel)

        data['a'] = a_idm

        new_v = np.maximum(0, data['v'] + data['a'] * dt)   # filter
        displacement = np.maximum(0, data['v'] * dt + data['a'] * np.square(dt) / 2)  # filter

        data['p'] += displacement
        data['v'] = new_v

    def lane_change(self):
        if self.ramp_active.iloc[0, 0]+2 > self.ramp_end:  # stop moving (head exceeds ramp end)
            self.ramp_active.iloc[0, 0:2] = self.ramp_end-2, 0   # fix position and speed

        ready2merge = self.ramp_active[(self.ramp_active['p']-2) > self.ramp_start]  # tail exceeds ramp start

        if len(ready2merge) > 0:    # only for car that can merge
            g_min = 1   # meter
            cl, cf = 0.9, 0.2

            r_cars_index = -1
            for index, car in ready2merge.iterrows():
                r_cars_index += 1
                leaders = self.main_active[self.main_active['p'] > car['p']]
                n_leaders = len(leaders)
                if n_leaders > 0:
                    leader = leaders.iloc[-1, :]
                    vl = leader['v']
                    pl = leader['p']
                else:
                    vl = 100
                    pl = 450

                follower = self.main_active[self.main_active['p'] < car['p']].iloc[0, :]

                pf = follower['p']
                vf = follower['v']
                vs = car['v']
                ps = car['p']

                g_l_min = g_min + np.maximum(0, cl*(vs-vl))
                g_f_min = g_min + np.maximum(0, cf*(vf-vs))

                # check merge
                if (pl - ps - 5 >= g_l_min) and (ps - pf - 5 >= g_f_min):
                    self.r_cars[r_cars_index].merged = True
                    self.main_active = self.main_active.append(car)   # + to main_active
                    self.main_active = self.main_active.sort_values('p', 0, ascending=False)  # sort main road
                    self.m_cars.insert(n_leaders, self.r_cars.pop(r_cars_index))    # add to m_car object list, remove from r_car object list
                    self.ramp_active = self.ramp_active[self.ramp_active['p'] != ps]  # remove car from ramp_active
                    r_cars_index -= 1

    def on_draw(self):
        self.clear()
        pyglet.gl.glClearColor(240, 240, 240, 255)
        self.fps.draw()

        self.batch_labels.draw()
        self.meter_light.vertex_list.draw(pyglet.gl.GL_QUADS)
        self.batch_car.draw()
        self.batch_road_line.draw()

    def on_key_press(self, symbol, modifiers):
        global SPEED_UP, MAIN_VOLUME, RAMP_VOLUME, AV_RATE

        if symbol == pyglet.window.key.Q:
            SPEED_UP += 0.5
            self.label_speed.text = 'Running speed (Q, A): %.1fx' % SPEED_UP
        elif symbol == pyglet.window.key.A:
            SPEED_UP -= 0.5
            self.label_speed.text = 'Running speed (Q, A): %.1fx' % SPEED_UP
        elif symbol == pyglet.window.key.W:
            MAIN_VOLUME += 50
            self.label_main_volume.text = 'Main road (W, S): %i (veh/h)' % MAIN_VOLUME
        elif symbol == pyglet.window.key.S:
            if MAIN_VOLUME > 50: MAIN_VOLUME -= 50
            self.label_main_volume.text = 'Main road (W, S): %i (veh/h)' % MAIN_VOLUME

        elif symbol == pyglet.window.key.E:
            RAMP_VOLUME += 50
            # self.meter_light.min_red_t = red_time()
            self.label_ramp_volume.text = 'Ramp (E, D): %i (veh/h)' % RAMP_VOLUME
            # self.label_light.text = 'Red light: %.1f (s)' % self.meter_light.min_red_t
        elif symbol == pyglet.window.key.D:
            if RAMP_VOLUME > 50: RAMP_VOLUME -= 50
            # self.meter_light.min_red_t = red_time()
            self.label_ramp_volume.text = 'Ramp (E, D): %i (veh/h)' % RAMP_VOLUME
            # self.label_light.text = 'Red light: %.1f (s)' % self.meter_light.min_red_t
        elif symbol == pyglet.window.key.R:
            if AV_RATE <= 0.9: AV_RATE += 0.1
            self.label_av_rate.text = 'AV rate (R, F): %.2f' % AV_RATE
        elif symbol == pyglet.window.key.F:
            if AV_RATE >= 0.1: AV_RATE -= 0.1
            self.label_av_rate.text = 'AV rate (R, F): %.2f' % AV_RATE

    def build_road(self, line_coords):
        self.batch_road_line = pyglet.graphics.Batch()
        self.road_width = road_width = int(scaling(3, 'm'))

        lc = line_coords[0]+line_coords[1]
        lc[1] -= road_width/2
        lc[3] -= road_width/2
        lc[2] = scaling(450-self.ramp_end, 'm')  # ramp entry end
        outer1_1 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', lc),
            ('c3i/static', (0, 0, 0) * 2)
        )
        lc = line_coords[0] + line_coords[1]
        lc[0] = scaling(450-self.ramp_start, 'm')  # ramp entry start
        lc[1] -= road_width / 2
        lc[2] = self.width
        lc[3] -= road_width / 2
        outer1_2 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', lc),
            ('c3i/static', (0, 0, 0) * 2)
        )
        lc = line_coords[2] + line_coords[3]    # top-most side
        lc[1] += road_width / 2
        lc[3] += road_width / 2
        outer2 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', lc),
            ('c3i/static', (0, 0, 0) * 2)
        )
        lc = line_coords[0] + line_coords[1]    # lowest side
        lc[1] += road_width / 2
        lc[3] += road_width / 2
        inner1 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', lc),
            ('c3i/static', (0, 0, 0) * 2)
        )
        lc = line_coords[2] + line_coords[3]    # top-inner side
        lc[1] -= road_width / 2
        lc[3] -= road_width / 2
        inner2 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', lc),
            ('c3i/static', (0, 0, 0) * 2)
        )
        # circle
        center = [line_coords[1][0], (line_coords[1][1]+line_coords[2][1])/2]

        radius = (line_coords[2][1]-line_coords[1][1] + road_width)/2
        lc = []
        for alpha in np.linspace(0, np.pi, 100):
            y = center[1] - radius * np.cos(alpha)
            x = center[0] + radius * np.sin(alpha)
            lc += [x, y]
            len_lc = len(lc)
            if len_lc >= 4:
                self.batch_road_line.add(
                    2, pyglet.gl.GL_LINES, None,
                    ('v2f/static', lc[len_lc-4: len_lc]),
                    ('c3i/static', (0, 0, 0) * 2)
                )
        radius = (line_coords[2][1] - line_coords[1][1] - road_width)/2
        lc = []
        for alpha in np.linspace(0, np.pi, 100):
            y = center[1] - radius * np.cos(alpha)
            x = center[0] + radius * np.sin(alpha)
            lc += [x, y]
            len_lc = len(lc)
            if len_lc >= 4:
                self.batch_road_line.add(
                    2, pyglet.gl.GL_LINES, None,
                    ('v2f/static', lc[len_lc-4: len_lc]),
                    ('c3i/static', (0, 0, 0) * 2)
                )

        ramp1 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', (line_coords[0][0]+scaling(450-self.ramp_end, 'm'), line_coords[0][1]-1/2*road_width,
                            line_coords[0][0]+scaling(450-self.ramp_end, 'm'), line_coords[0][1]-3/2*road_width)),
            ('c3i/static', (0, 0, 0) * 2)
        )
        ramp2 = self.batch_road_line.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f/static', (line_coords[0][0] + scaling(450-self.ramp_end, 'm'), line_coords[0][1] - 3 / 2 * road_width,
                            self.width, line_coords[0][1] - 3 / 2 * road_width)),
            ('c3i/static', (0, 0, 0) * 2)
        )

if __name__ == "__main__":
    np.random.seed(2)
    DESIRED_SPEED = 100
    MAIN_VOLUME = 1800
    RAMP_VOLUME = 600
    SPEED_UP = 8
    AV_RATE = 0.2

    GREEN = 4
    RED = red_time()


    win = Window()
    pyglet.app.run()