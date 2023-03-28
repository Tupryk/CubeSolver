from enviroment import CubeEnv
import torch

env = CubeEnv()
env.render()
env.step([1, 0, 1])
env.step([0, 1, 0])
env.step([0, 0, 2])
env.step([1, 1, 2])
env.step([2, 0, 1])
env.step([2, 0, 0])

env.step([2, 1, 0])
env.step([2, 1, 1])
env.step([1, 0, 2])
env.step([0, 1, 2])
env.step([0, 0, 0])
env.step([1, 1, 1])

env.render()

env.reset()
env.render()

"""
for i in range(100):
    axis = (action // (2*self.dim)) % 3
    direction = action % 2
    line = (action // 2) % self.dim
    print(torch.randint(3*2*3, (1,)))
"""
