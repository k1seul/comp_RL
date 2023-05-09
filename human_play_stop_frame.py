

from BallCatch import BallCatch 
import pygame 


terminated = False
truncated = False 

env = BallCatch(render_mode="human") 

for trial_num in range(9):
    state, info = env.reset()
    terminated = False
    truncated = False 

    pygame.event.clear()
    sum_reward = 0

    while not(terminated or truncated):
        

        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_LEFT:
                    action = 1
                elif ev.key == pygame.K_RIGHT:
                    action = 2
                elif ev.key == pygame.K_SPACE:
                    action = 0


                obs, reward, done, truncated, info = env.step(action)
                print(reward)
                sum_reward = sum_reward + reward

        
                if done or truncated:
                    print("Game over! Final score: {}".format(sum_reward))
                    terminated = done 

                    break 



