import pygame, Box2D
import numpy as np
import cv2


def center_of_mass(point_list):
    center = np.zeros((2,))
    for point in point_list:
        center += point
    return center / len(point_list)


class Renderer(object):
    """ This the abstract base class for rendering the simulated physical world. """

    # Box2D deals with meters, but we want to display pixels,
    # so define a conversion factor:
    PPM = 1.0  # pixels per meter

    def __init__(self, screen_width, screen_height):
        # self.world = world
        self.screen_width_px = screen_width
        self.screen_height_px = screen_height
        self.screen = None
        self.__visible = True

    @staticmethod
    def pixels_to_meters(pixels):
        return pixels / Renderer.PPM

    @staticmethod
    def meters_to_pixels(meters):
        return int(meters * Renderer.PPM)

    @property
    def is_visible(self):
        return self.__visible

    @is_visible.setter
    def is_visible(self, visible):
        self.__visible = visible

    def to_world_frame(self, point):
        return Renderer.pixels_to_meters(point[0]), Renderer.pixels_to_meters(self.screen_height_px - point[1])

    def to_screen_frame(self, point):
        return Renderer.meters_to_pixels(point[0]), self.screen_height_px - Renderer.meters_to_pixels(point[1])

    def get_frame(self, world):
        pass

    def reset(self):
        pass


class VideoRenderer(Renderer):
    """ This class has the role of rendering the simulated physical world. """

    COLOR_WHITE = (255, 255, 255, 255)
    COLOR_BLACK = (0, 0, 0, 0)

    def __init__(self, screen_width, screen_height):
        super(VideoRenderer, self).__init__(screen_width, screen_height)
        self.screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)
        pygame.display.set_caption('Simple pygame example')

    @Renderer.is_visible.setter
    def is_visible(self, visible):
        if not visible:
            pygame.display.iconify()
        Renderer.is_visible.fset(self, visible)

    def get_frame(self, world):
        self.reset()

        # Draw the world
        for body in world.bodies:
            # The body gives us the position and angle of its shapes
            if body.userData.visible:
                for fixture in body.fixtures:
                    # The fixture holds information like density and friction,
                    # and also the shape.
                    shape = fixture.shape

                    if isinstance(shape, Box2D.b2PolygonShape):
                        vertices = [self.to_screen_frame(body.transform * v) for v in shape.vertices]
                        pygame.draw.polygon(self.screen, self.COLOR_WHITE, vertices)
                    elif isinstance(shape, Box2D.b2EdgeShape):
                        vertices = [self.to_screen_frame(body.transform * v) for v in shape.vertices]
                        pygame.draw.line(self.screen, self.COLOR_WHITE, vertices[0], vertices[1])
                    elif isinstance(shape, Box2D.b2CircleShape):
                        center = self.to_screen_frame(body.position)
                        pygame.draw.circle(self.screen, self.COLOR_WHITE, center, self.meters_to_pixels(shape.radius))

        if self.is_visible:
            pygame.display.flip()

        array = np.frombuffer(self.screen.get_buffer(), dtype='uint8')
        image = np.reshape(array, (self.screen_height_px, self.screen_width_px, 4))
        return np.reshape(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (self.screen_height_px, self.screen_width_px,1))

    def reset(self):
        self.screen.fill(self.COLOR_BLACK)


class CentroidVideoRenderer(Renderer):

    def __init__(self, screen_width, screen_height):
        super(CentroidVideoRenderer, self).__init__(screen_width, screen_height)
        self.downsampling = 0

    def get_frame(self, world):

        if self.downsampling == 0:
            self.downsampling = self.__compute_downsampling_factor(world)

        self.reset()

        # Draw the world
        for body in world.bodies:
            # The body gives us the position and angle of its shapes
            if body.userData.visible:
                for fixture in body.fixtures:
                    # The fixture holds information like density and friction,
                    # and also the shape.
                    shape = fixture.shape

                    if isinstance(shape, Box2D.b2CircleShape):
                        center = self.to_screen_frame(body.position)
                        center = self.__fit_in_screen(center)
                        self.screen[center[0], center[1]] = 255
                    elif isinstance(shape, Box2D.b2PolygonShape) or isinstance(shape, Box2D.b2EdgeShape):
                        vertices = [body.transform * v for v in shape.vertices]
                        center = self.to_screen_frame(center_of_mass(vertices))
                        center = self.__fit_in_screen(center)
                        self.screen[center[0], center[1]] = 255

        return self.screen

    def __compute_downsampling_factor(self, world):
        # for body in world.bodies:
        #  if body.userData.visible:
        #     for fixture in body.fixtures:
        #        shape = fixture.shape
        #        if isinstance(shape, Box2D.b2CircleShape):
        #           return 1.0 / self.meters_to_pixels(2*shape.radius)
        # raise LookupError("World contains no circle!")
        return 1.0 / 8.0

    def __fit_in_screen(self, point):
        x, y = point
        if x < 0:
            x = 0
        if x >= int(self.screen_height_px * self.downsampling):
            x = int(self.screen_height_px * self.downsampling) - 1
        if y < 0:
            y = 0
        if y >= int(self.screen_width_px * self.downsampling):
            y = int(self.screen_width_px * self.downsampling) - 1
        return x, y

    def to_screen_frame(self, point):
        point = super(CentroidVideoRenderer, self).to_screen_frame(point)
        return int(point[1] * self.downsampling), int(point[0] * self.downsampling)

    def reset(self):
        if self.downsampling == 0:
            return
        self.screen = np.zeros((int(self.screen_height_px * self.downsampling),
                                int(self.screen_width_px * self.downsampling)), dtype='uint8')


class PositionAndVelocityExtractor(Renderer):

    def __init__(self, screen_width, screen_height):
        super(PositionAndVelocityExtractor, self).__init__(screen_width, screen_height)
        self.screen = np.zeros((0, 4), dtype=np.float32)
        self.is_visible = False

    def get_frame(self, world):
        """ FIXME! This function works only in some specific conditions """
        self.reset()

        body_index = 0
        num_of_objects = len(world.bodies) - 1
        self.screen = np.zeros((num_of_objects * 4,), dtype=np.float32)
        for body in world.bodies:
            # The body gives us the position and angle of its shapes
            if body.userData.visible:
                for fixture in body.fixtures:
                    # The fixture holds information like density, friction and shape.
                    shape = fixture.shape

                    if isinstance(shape, (Box2D.b2PolygonShape, Box2D.b2CircleShape)):
                        if isinstance(shape, Box2D.b2PolygonShape):
                            vertices = [self.to_screen_frame(body.transform * v) for v in shape.vertices]
                            position = center_of_mass(vertices)
                        else:
                            position = self.to_screen_frame(body.position)
                        velocity = body.linearVelocity
                        self.screen[body_index:body_index + 4] = np.array(
                            [position[0], position[1], velocity[0], velocity[1]])
                        body_index += 4

        return self.screen

    def reset(self):
        self.screen = np.zeros_like(self.screen)
