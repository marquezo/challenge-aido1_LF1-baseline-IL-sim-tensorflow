import numpy as np
from tqdm import tqdm
from gym_duckietown.envs import DuckietownEnv

from _loggers import Reader
from model import TensorflowModel


env = DuckietownEnv(
    map_name='loop_empty',
    max_steps=500001,
    domain_rand=False,
    camera_width=80,
    camera_height=60,
    accept_start_angle_deg=4,  # start close to straight
    full_transparency=True,
)


def evaluate(env, episodes=5, steps=500):

    observation = env.reset()

    # we can use the gym reward to get an idea of the performance of our model
    cumulative_reward = 0.0

    for episode in range(0, episodes):
        for steps in range(0, steps):
            action = model.predict(observation)
            #action = np.abs(action)
            observation, reward, done, info = env.step(action)
            env.render()
            cumulative_reward += reward

            if done:
                break
            # env.render()
        # we reset after each episode, or not, this really depends on you
        env.reset()

    print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // episodes))

    return cumulative_reward // episodes


# configuration zone
BATCH_SIZE = 64
EPOCHS = 20
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "trained_models/behavioral_cloning"

reader = Reader('train.log')

observations, actions = reader.read()
actions = np.array(actions)
observations = np.array(observations)

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

# we trained for EPOCHS epochs
# TODO: need a validation set: use the reward to stop training
epochs_bar = tqdm(range(EPOCHS))
for i in epochs_bar:
    # we defined the batch size, this can be adjusted according to your computing resources...
    loss = None
    for batch in range(0, len(observations), BATCH_SIZE):
        loss = model.train(
            observations=observations[batch:batch + BATCH_SIZE],
            actions=actions[batch:batch + BATCH_SIZE]
        )

    valid_reward = evaluate(env)
    epochs_bar.set_postfix({'loss': loss, 'valid_reward': valid_reward})

    # every 10 epochs, we store the model we have
    # but I'm sure that you're smarter than that, what if this model is worse than the one we had before
    if i % 10 == 0:
        model.commit()
        epochs_bar.set_description('Model saved...')
    else:
        epochs_bar.set_description('')


# the loss at this point should be on the order of 2e-2, which is far for great, right?

# we release the resources...
model.close()
reader.close()

