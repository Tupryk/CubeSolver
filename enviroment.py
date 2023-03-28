from collections import Counter
from random import randrange

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete, Tuple
from termcolor import colored

colored_numbers = {
    0: colored(' #', 'red'),
    1: colored(' #', 'yellow'),
    2: colored(' #', 'green'),
    3: colored(' #', 'blue'),
    4: colored(' #', 'white'),
    5: colored(' #', 'magenta')
}

class CubeEnv(Env):
    def __init__(self, dim=3, shuffle_steps=1):

        faces_count = 6

        faces = []
        for i in range(faces_count):
            new_face = np.full( (dim, dim), i )
            faces.append(new_face)

        self.state = np.array(faces)
        self.step_count = 0
        self.max_steps = shuffle_steps+1
        self.dim = dim

        self.shuffle_steps = shuffle_steps

        # Rotation dimension: 3 (x, y, z)
        # Rotation directions: 2 (clock-wise, counter-clock-wise)
        # Rotation lines per face: dim
        self.action_space = Tuple([
                                Discrete(3),  # Angle of rotation
                                Discrete(2),   # Direction of rotation
                                Discrete(3),   # Line on which to perform the action
                            ])
        self.max_score = self.score()
        flat_dim = faces_count*dim*dim
        self.observation_space = Box(low=np.zeros((flat_dim)), high=np.full((flat_dim), faces_count-1), dtype=np.float64)

    def solved(self):
        for face in range(6):
            starter = self.state[face, 0, 0]
            for i in range(1, self.dim):
                for j in range(1, self.dim):
                    if self.state[face, i, j] != starter:
                        return False
        return True
    
    def score(self):
        get_score = 0
        for face in self.state:
            convo = Counter(face.flatten())
            convo = convo.most_common(1)[0][1]
            if convo > 5:
                get_score += convo
        return get_score
    
    def make_move(self, axis, direction, line):
        # Turns out programming the logic for a rubiks cube is harder than I expected... but it works!!!
        if axis == 0: # x-axis
            # Rotate faces 1 and 3
            if direction == 1: # couter-clockwise
                tmp = np.copy(self.state[2, line, :])
                self.state[2, line, :] = np.copy(self.state[0, line, :])
                self.state[0, line, :] = np.flip(np.copy(self.state[4, (self.dim-1)-line, :]))
                self.state[4, (self.dim-1)-line, :] = np.flip(np.copy(self.state[5, line, :]))
                self.state[5, line, :] = tmp
            else:
                tmp = np.copy(self.state[2, line, :])
                self.state[2, line, :] = np.copy(self.state[5, line, :])
                self.state[5, line, :] = np.flip(np.copy(self.state[4, (self.dim-1)-line, :]))
                self.state[4, (self.dim-1)-line, :] = np.flip(np.copy(self.state[0, line, :]))
                self.state[0, line, :] = tmp

            if line == 0: # Rotate face 1
                self.state[1] = np.copy(np.rot90(self.state[1])) if direction == 1 else np.copy(np.rot90(self.state[1], 3))
            elif line == self.dim-1: # Rotate face 3
                self.state[3] = np.copy(np.rot90(self.state[3])) if direction == 0 else np.copy(np.rot90(self.state[3], 3))

        elif axis == 1: # y-axis
            # Rotate faces 2 and 4
            if direction == 1: # couter-clockwise
                tmp = np.copy(self.state[3, line, :])
                self.state[3, line, :] = np.copy(self.state[0, :, (self.dim-1)-line])
                self.state[0, :, (self.dim-1)-line] = np.flip(np.copy(self.state[1, (self.dim-1)-line, :]))
                self.state[1, (self.dim-1)-line, :] = np.copy(self.state[5, :, line])
                self.state[5, :, line] = np.flip(tmp)
            else:
                tmp = np.copy(self.state[3, line, :])
                self.state[3, line, :] = np.flip(np.copy(self.state[5, :, line]))
                self.state[5, :, line] = np.copy(self.state[1, (self.dim-1)-line, :])
                self.state[1, (self.dim-1)-line, :] = np.flip(np.copy(self.state[0, :, (self.dim-1)-line]))
                self.state[0, :, (self.dim-1)-line] = tmp

            if line == 0: # Rotate face 2
                self.state[2] = np.copy(np.rot90(self.state[2])) if direction == 1 else np.copy(np.rot90(self.state[2], 3))
            elif line == self.dim-1: # Rotate face 4
                self.state[4] = np.copy(np.rot90(self.state[4])) if direction == 0 else np.copy(np.rot90(self.state[4], 3))

        elif axis == 2: # z-axis
            # Rotate faces 0 and 5
            if direction == 1: # couter-clockwise
                tmp = np.copy(self.state[2, :, line])
                self.state[2, :, line] = np.copy(self.state[3, :, line])
                self.state[3, :, line] = np.copy(self.state[4, :, line])
                self.state[4, :, line] = np.copy(self.state[1, :, line])
                self.state[1, :, line] = tmp
            else:
                tmp = np.copy(self.state[2, :, line])
                self.state[2, :, line] = np.copy(self.state[1, :, line])
                self.state[1, :, line] = np.copy(self.state[4, :, line])
                self.state[4, :, line] = np.copy(self.state[3, :, line])
                self.state[3, :, line] = tmp
            
            if line == 0: # Rotate face 0
                self.state[0] = np.copy(np.rot90(self.state[0])) if direction == 1 else np.copy(np.rot90(self.state[0], 3))
            elif line == self.dim-1: # Rotate face 5
                self.state[5] = np.copy(np.rot90(self.state[5])) if direction == 0 else np.copy(np.rot90(self.state[5], 3))

        
    def step(self, action):

        self.step_count += 1

        # Update state
        axis = (action // (2*self.dim)) % 3
        direction = action % 2
        line = (action // 2) % self.dim
        self.make_move(axis, direction, line)

        # Respond to agent
        solved = self.solved()
        # reward = self.score()
        done = False
        truncated = False
        if self.step_count >= self.max_steps or solved:
            done = True
            truncated = not solved
        reward = 1 if done and not truncated else -1
        info = {}

        return self.state.flatten(), reward, done, truncated, info

    def render(self):
        for j in range(self.dim):
            print("  "*self.dim, end="")
            for i in range(self.dim):
                print(colored_numbers[self.state[0, i, j]], end="")
            print()

        for a in range(self.dim):
            for b in range(1, 5):
                for c in range(self.dim):
                    print(colored_numbers[self.state[b, c, a]], end="")
            print()

        for j in range(self.dim):
            print("  "*self.dim, end="")
            for i in range(self.dim):
                print(colored_numbers[self.state[5, i, j]], end="")
            print()

        if self.solved():
            print("The cube is solved!!!")
        else:
            print("The cube is not solved!!!")
    
    def reset(self, seed=1):
        faces_count = 6
        faces = []
        for i in range(faces_count):
            new_face = np.full( (self.dim, self.dim), i )
            faces.append(new_face)
        self.state = np.array(faces)
        self.step_count = 0

        # Shuffle cube
        for i in range(self.shuffle_steps):
            self.make_move(randrange(3), randrange(2), randrange(self.dim))

        info = {}
        return self.state.flatten(), info
    