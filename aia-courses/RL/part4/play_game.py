import pygame # 1.9.3
import sys
sys.path.append("game/")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import wrapped_flappy_bird as game
import numpy as np
import time

# press space key to flap
def play_by_human():
    play_times = 5
    while play_times > 0:
        game_state = game.GameState()
        game_state.acceleration = False
        continue_playing = True
        score = 0

        previous_time = time.time()
        while continue_playing:
            a_t = np.zeros([2])
            keystate = pygame.key.get_pressed()
            if keystate[pygame.K_SPACE]:
                a_t = np.array([0,1])
            else:
                a_t = np.array([1,0])
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            score += r_t
            if time.time()-previous_time>1:
                previous_time = time.time()
                print('current score:',score)
            if terminal:
                print('total score:',score)
                continue_playing = False
                play_times -= 1
if __name__ == "__main__":
    play_by_human()