import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Create_Game(tk.Tk, object):
    def __init__(self):
        super(Create_Game, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('Mario')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='black',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        #hell1_center = origin + np.array([UNIT * 2, UNIT])
        #self.hell1 = self.canvas.create_rectangle(
        #    hell1_center[0] - 15, hell1_center[1] - 15,
        #    hell1_center[0] + 15, hell1_center[1] + 15,
        #    fill='black')
        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin.copy() 
        oval_center[0]+= UNIT * 3
        oval_center[1]+= UNIT * 3
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='green')
        
        self.image_file = tk.PhotoImage(file='environment/tube.png')
        self.image = self.canvas.create_image(
            oval_center[0], oval_center[1],
            image = self.image_file)
        
        # create Mario

        self.image_file1 = tk.PhotoImage(file='environment/mario.png')
        self.rect = self.canvas.create_image(
            origin[0], origin[1],
            image = self.image_file1)
        

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        # create Mario
        self.image_file1 = tk.PhotoImage(file='environment/mario.png')
        self.rect = self.canvas.create_image(
            origin[0], origin[1],
            image = self.image_file1)
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        """        

        if action == 0:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 1:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT        
        """       
        
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.image):
            reward = 1
            done = True
        #elif next_coords in [self.canvas.coords(self.hell1)]:
        #    reward = -1
        #    done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()
        
    def close(self):
        self.quit()
