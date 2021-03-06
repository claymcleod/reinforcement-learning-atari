from skimage.color import rgb2gray
from skimage.transform import resize

TEST_ENV_NAME = "MsPacman-v0"

def dimensions_for_env(env_name):
    if env_name == "MsPacman-v0":
        return {
            'resize_to': (118, 90),
            'network_in_x': 90,
            'network_in_y': 96,
            'start': 0,
            'stop': 96
        }
    elif env_name == "SpaceInvaders-v0":
        return {
            'resize_to': (118, 90),
            'network_in_x': 90,
            'network_in_y': 118,
            'start': 0,
            'stop': 118
        }
    elif env_name == "Pong-v0":
        return {
            'resize_to': (110, 84),
            'network_in_x': 84,
            'network_in_y': 84,
            'start': 13,
            'stop': 97
        }
    elif env_name == "Breakout-v0":
        return {
            'resize_to': (110, 84),
            'network_in_x': 84,
            'network_in_y': 110,
            'start': 0,
            'stop': 110
        }
    else:
        raise RuntimeError("Dimensions not implemented for {}".format(env_name))

def _preprocess_img(observation_t, env_name):
    dimensions = dimensions_for_env(env_name)
    return resize(rgb2gray(observation_t), dimensions["resize_to"])[dimensions["start"]:dimensions["stop"], :]

if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    env = gym.make(TEST_ENV_NAME)
    x_t = env.reset()
    for i in range(50):
        x_t, reward_t, done, _ = env.step(0)
    x_t = _preprocess_img(x_t, TEST_ENV_NAME)
    print(x_t.shape)
    plt.imshow(x_t)
    plt.show()
