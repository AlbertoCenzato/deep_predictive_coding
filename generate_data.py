#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, os
import numpy as np
from random import random, randint
from bouncing_balls.bouncing_balls import BouncingBalls
from bouncing_balls.video_writer import BufferedBinaryWriter
from bouncing_balls.render import VideoRenderer, PositionAndVelocityExtractor, CentroidVideoRenderer
from deep_predictive_coding.model_type import ModelType
from settings import TESTS_CONFIG

# ------------------ Constants ----------------------------------
screen_width = 64
screen_height = 48
train = "train"
test = "test"


# ------------------ Function definitions ------------------------

def get_angle(vect1, vect2):
    """Returns the angle (in radians) between two given vectors"""
    return math.acos(np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2)))


def random_trajectory_through_rectangle(init_pos, rect):
    """Returns a random unit norm vector defining the trajectory
      that starts from point initPos and intersects with the rectangle rect"""

    min_x, max_x = rect[0] + 10, rect[0] + rect[2] - 10
    min_y, max_y = rect[1] + 10, rect[1] + rect[3] - 10
    vertices = [(min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y)]

    if init_pos[0] < min_x:
        if init_pos[1] < min_y:
            vertex1, vertex2 = vertices[1], vertices[3]
        elif init_pos[1] > max_y:
            vertex1, vertex2 = vertices[0], vertices[2]
        else:
            vertex1, vertex2 = vertices[0], vertices[1]
    elif init_pos[0] > max_x:
        if init_pos[1] < min_y:
            vertex1, vertex2 = vertices[0], vertices[2]
        elif init_pos[1] > max_y:
            vertex1, vertex2 = vertices[1], vertices[3]
        else:
            vertex1, vertex2 = vertices[2], vertices[3]
    else:
        if init_pos[1] < min_y:
            vertex1, vertex2 = vertices[0], vertices[3]
        elif init_pos[1] > max_y:
            vertex1, vertex2 = vertices[1], vertices[2]

    vect1 = (vertex1[0] - init_pos[0], vertex1[1] - init_pos[1])  # work with a coordinate frame centered in initPos
    vect2 = (vertex2[0] - init_pos[0], vertex2[1] - init_pos[1])  # work with a coordinate frame centered in initPos
    alpha1 = get_angle((1, 0), vect1)
    if vect1[1] < 0:
        alpha1 = -alpha1
    alpha2 = get_angle((1, 0), vect2)
    if vect2[1] < 0:
        alpha2 = -alpha2
    alpha12 = get_angle(vect1, vect2)

    rand_angle = random() * alpha12
    if alpha1 > alpha2:
        alpha1, alpha2 = alpha2, alpha1
    if alpha1 < -math.pi / 2 and alpha2 > math.pi / 2:
        rand_angle -= alpha1
    else:
        rand_angle += alpha1

    return (math.cos(rand_angle), math.sin(rand_angle))


def random_pos_outside_rectangle(rect):
    """Returns a random position, (x,y) tuple, such that (x,y)
      is not a point inside the rectangle area"""
    tol = 3
    x = random() * screen_width
    if rect[0] - tol < x < rect[0] + rect[2] + tol:
        empty_interval_y = screen_height - rect[3] - 2 * tol
        y = random() * empty_interval_y
        if y > rect[1] - tol:
            y += rect[3] + 2 * tol
    else:
        y = random() * screen_height
    return x, y


def generate_dataset(config, train_or_test, suppress_output):
    if train_or_test != train and train_or_test != test:
        raise ValueError("trainOrTest parameter must have one of the two values 'test' or 'train'")
    dataset_dir = os.path.join(config.data_dir, train_or_test)
    if not os.path.exists(dataset_dir): os.mkdir(dataset_dir)
    file_name = os.path.join(dataset_dir, "bouncing_balls_")
    extension = ".npy"

    rectangle = (22, 19, 20, 10)
    fps = 60
    samples = config.sequences if train_or_test == train else config.sequences // 10
    frame_per_sequence = config.sequence_len

    if config.model_type == ModelType.SINGLE_PIXEL_ACTIVATION:
        renderer = CentroidVideoRenderer(screen_width, screen_height)
    elif config.model_type == ModelType.STATE_VECTOR:
        renderer = PositionAndVelocityExtractor(screen_width, screen_height)
    else:
        renderer = VideoRenderer(screen_width, screen_height)

    with BouncingBalls(renderer) as bouncing_balls:
        bouncing_balls.set_fps(fps)
        bouncing_balls.suppress_output(suppress_output)
        writer = BufferedBinaryWriter()
        for i in range(samples):
            # print("Rendering sequence " + str(i))
            bouncing_balls.save_to(file_name + str(i) + extension, writer)

            if config.occlusion:
                bouncing_balls.add_rectangular_occlusion(rectangle)
                for j in range(config.balls):
                    vel = np.random.normal(config.mean_vel, config.mean_vel // 10)
                    position = random_pos_outside_rectangle(rectangle)
                    (vx, vy) = random_trajectory_through_rectangle(position, rectangle)
                    bouncing_balls.add_circle(position, (vx * vel, vy * vel))
            else:
                for j in range(config.balls):
                    if config.dof == 1:
                        bouncing_balls.add_circle(bouncing_balls.get_rand_pos(), (2 * 5000 * random() - 5000, 0))
                    elif config.dof == 2:
                        bouncing_balls.add_rand_circle(max_vel=config.mean_vel)

            for k in range(frame_per_sequence):
                bouncing_balls.step()

            bouncing_balls.reset()


def generate_autoencoder_dataset(dataset_dir, samples):
    if not os.path.exists(dataset_dir): os.mkdir(dataset_dir)
    file_name = os.path.join(dataset_dir, "bouncing_balls_")
    extension = ".npy"

    fps = 60
    frame_per_sequence = 1
    with BouncingBalls(screen_width, screen_height) as bouncing_balls:
        bouncing_balls.set_fps(fps)
        bouncing_balls.suppress_output(True)
        writer = BufferedBinaryWriter()
        for i in range(samples):
            print("Rendering sequence " + str(i))
            bouncing_balls.save_to(file_name + str(i) + extension, writer)

            for j in range(randint(1, 5)):
                bouncing_balls.add_rand_circle(max_vel=5000)

            for k in range(frame_per_sequence):
                bouncing_balls.step()

            bouncing_balls.reset()


def generate(config, suppress_output=True):
    if os.path.exists(config.data_dir):  # if dataset already generated do nothing
        train_dir = os.path.join(config.data_dir, train)
        test_dir = os.path.join(config.data_dir, test)
        if len(os.listdir(train_dir)) == config.sequences and len(os.listdir(test_dir)) == config.sequences // 10:
            return
    else:
        os.mkdir(config.data_dir)

    generate_dataset(config, train, suppress_output)
    generate_dataset(config, test, suppress_output)


if __name__ == "__main__":
    # generateAutoencoderDataset("C:\\Users\\Alberto\\Projects\\Datasets\\bouncing_balls\\autoencoder_dataset\\test", 100)

    for config in TESTS_CONFIG:
        print("Generating dataset " + str(config))
        generate(config)  # , False)
        print("\n\n")
