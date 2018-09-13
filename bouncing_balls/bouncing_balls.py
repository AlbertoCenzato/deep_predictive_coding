from __future__ import division

import pygame, math
import Box2D
from numpy import random


def get_rand_vel(mean_vel):
    angle = 2 * math.pi * random.random()
    norm = random.normal(mean_vel, mean_vel // 10)
    vx = math.cos(angle) * norm
    vy = math.sin(angle) * norm
    return vx, vy


class BouncingBalls(object):
    # --- constants ---
    DEFAULT_FPS = 60
    SAVE_VIDEO = 'save_video'
    SAVE_STATE = 'save_state'

    class BodyData(object):

        def __init__(self, name, visible=True):
            self.name = name
            self.visible = visible

    def __init__(self, renderer):
        """ Instantiates a dynamic bouncing balls model.
          operating_mode: one of 'save_video' or 'save_state' """
        self.__fps = 0
        self.__timeStep = 0
        self.bounding_box = None
        self.writer = None

        self.save = False
        self.set_fps(self.DEFAULT_FPS)

        # --- pybox2D setup ---
        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.clock = pygame.time.Clock()

        # --- pygame setup ---
        self.screenOutput = True
        self.renderer = renderer
        self.screenWidth_m = self.renderer.pixels_to_meters(self.renderer.screen_width_px)
        self.screenHeight_m = self.renderer.pixels_to_meters(self.renderer.screen_height_px)

        self.create_screen_bounding_box()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.quit()


    def set_fps(self, fps):
        self.__fps = fps
        self.__timeStep = 1.0 / fps


    def get_rand_pos(self):
        return random.random() * self.renderer.screen_width_px, random.random() * self.renderer.screen_height_px


    def create_screen_bounding_box(self):
        screen_center = (self.screenWidth_m / 2, self.screenHeight_m / 2)
        bottom_left   = (-screen_center[0], -screen_center[1])
        bottom_right  = ( screen_center[0], -screen_center[1])
        top_right     = ( screen_center[0],  screen_center[1])
        top_left      = (-screen_center[0],  screen_center[1])

        self.bounding_box = self.world.CreateStaticBody(position=screen_center,
                                                        shapes=[Box2D.b2EdgeShape(vertices=[bottom_left,  bottom_right]),
                                                                Box2D.b2EdgeShape(vertices=[bottom_right, top_right]),
                                                                Box2D.b2EdgeShape(vertices=[top_right,    top_left]),
                                                                Box2D.b2EdgeShape(vertices=[top_left,     bottom_left])])
        self.bounding_box.userData = BouncingBalls.BodyData("world_bounding_box", visible=False)


    def enable_bounding_box(self, enable):
        self.bounding_box.active = enable


    def suppress_output(self, no_output):
        """ Runs the simulation at maximum allowable speed without visualization """
        self.renderer.is_visible = not no_output


    def save_to(self, path, writer):
        self.save = True
        self.writer = writer
        self.writer.FPS = self.__fps
        self.writer.resolution = (self.renderer.screen_width_px, self.renderer.screen_height_px)
        self.writer.open(path)


    def add_line(self, p1, p2):
        p1 = self.renderer.to_world_frame(p1)
        p2 = self.renderer.to_world_frame(p2)
        m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        p1 = (p1[0] - m[0], p1[1] - m[1])  # compute p1 coordinates wrt m
        p2 = (p2[0] - m[0], p2[1] - m[1])  # compute p2 coordinates wrt m
        line = self.world.CreateStaticBody(position=m, shapes=[Box2D.b2EdgeShape(vertices=[p1, p2])])
        line.active = False
        line.userData = BouncingBalls.BodyData("line", visible=False)
        print("Warning line rendering disabled!")


    def add_rectangular_occlusion(self, rect):
        width_px, height_px = rect[2], rect[3]
        center_px = (rect[0] + width_px / 2, rect[1] + height_px / 2)
        width_m, height_m = self.renderer.pixels_to_meters(width_px), self.renderer.pixels_to_meters(height_px)
        occlusion = self.world.CreateStaticBody(position=self.renderer.to_world_frame(center_px))
        occlusion.CreatePolygonFixture(box=(width_m / 2, height_m / 2), density=0, friction=0, restitution=0)
        occlusion.active = False
        occlusion.userData = BouncingBalls.BodyData("rectangular_occlusion")


    def add_square(self, pos, vel, angle):
        pos = self.renderer.to_world_frame(pos)
        vel = (self.renderer.pixels_to_meters(vel[0]), -self.renderer.pixels_to_meters(vel[1]))
        size = 1
        center = (pos[0] + size / 2, pos[1] + size / 2)
        square = self.world.CreateDynamicBody(position=center, angle=angle)
        square.CreatePolygonFixture(box=(size, size), density=1, friction=0, restitution=1)
        square.linearVelocity = vel
        square.userData = BouncingBalls.BodyData("square")


    def add_rand_square(self):
        position = self.get_rand_pos()
        velocity = get_rand_vel(10)
        angle = random.random() * 90
        self.add_square(position, velocity, angle)


    def add_circle(self, pos, vel):
        pos = self.renderer.to_world_frame(pos)
        vel = (self.renderer.pixels_to_meters(vel[0]), -self.renderer.pixels_to_meters(vel[1]))
        circle = self.world.CreateDynamicBody(position=pos)
        fixture_def = Box2D.b2FixtureDef(shape=(Box2D.b2CircleShape(radius=5)), density=1, friction=0, restitution=1)
        circle.CreateFixture(fixture_def)
        circle.linearVelocity = vel
        circle.userData = BouncingBalls.BodyData("circle")


    def add_rand_circle(self, max_vel=5000):
        position = self.get_rand_pos()
        velocity = get_rand_vel(max_vel)
        self.add_circle(position, velocity)


    def step(self):
        frame = self.renderer.get_frame(self.world)
        if self.save:
            self.writer.write(frame)

        # Make Box2D simulate the physics of our world for one step.
        # Instruct the world to perform a single step of simulation.
        self.world.Step(self.__timeStep, 10, 10)

        # Try to keep at the target FPS
        if self.renderer.is_visible:
            self.clock.tick(self.__fps)
            return self.clock.get_time()

        # self.clock.tick()
        return self.__timeStep * 1000


    def run_simulation(self, time_s):
        elapsed_ms = 0
        max_time_ms = time_s * 1000
        while elapsed_ms < max_time_ms:
            elapsed_ms += self.step()

        if self.save:
            self.writer.close()


    def reset(self):
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        self.create_screen_bounding_box()
        if self.save:
            self.writer.close()
        self.clock = pygame.time.Clock()
        self.renderer.reset()


    def quit(self):
        pygame.quit()
        if self.save:
            self.writer.close()