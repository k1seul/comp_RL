from BallCatch import BallCatch
import pygame


env = BallCatch(render_mode='human')

# Set up the player move timer
player_move_time = 0
player_move_delay = 500 # milliseconds

for episode_num in range(1,10):
    
    state, info = env.reset()
    terminated = False
    truncated = False 

    while not(terminated or truncated):

            # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Quit the game
                pygame.quit()
                quit()

        current_time = pygame.time.get_ticks()

        if current_time - player_move_time >= player_move_delay:
            # Get an action from the player
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 1
                player_move_time = current_time
            elif keys[pygame.K_RIGHT]:
                action = 2
                player_move_time = current_time
            else:
                action = 0


            # Take the action and observe the new state

            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                print("Game over! Final score: {}".format(reward))

                break 



        
