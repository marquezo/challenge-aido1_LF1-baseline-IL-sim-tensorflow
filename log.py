import cv2
from gym_duckietown.envs import DuckietownEnv
from teacher import PurePursuitExpert
from _loggers import Logger
import numpy as np

# Log configuration, you can pick your own values here
# the more the better? or the smarter the better?
EPISODES = 50
STEPS = 512 * 2

DEBUG = False
starting_tiles = ["14", "13", "25", "35"]

env = DuckietownEnv(
    map_name='loop_empty',  # check the Duckietown Gym documentation, there are many maps of different complexity
    max_steps=EPISODES * STEPS,
    user_tile_start = starting_tiles[np.random.choice(len(starting_tiles))],
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4,  # start close to straight
    full_transparency=True,
)

# this is an imperfect demonstrator... I'm sure you can construct a better one.
expert = PurePursuitExpert(env=env)

# please notice
logger = Logger(env, log_file='train.log')

# let's collect our samples
for episode in range(0, EPISODES):
    print("Starting episode {}".format(episode))

    for steps in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        action = expert.predict(steps, None)

        if action is None:
            print("Invalid action: Ending the episode")
            break

        observation, reward, done, info = env.step(action)
        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)

        logger.log(observation, action, reward, done, info)
        env.render()
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here
    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()
    env.user_tile_start = starting_tiles[np.random.choice(len(starting_tiles))]

# we flush everything and close the file, it should be ~ 120mb
# NOTICE: we make the log file read-only, this prevent us from erasing all collected data by mistake
# believe me, this is an important issue... can you imagine loosing 2 GB of data? No? We do...
logger.close()
env.close()
