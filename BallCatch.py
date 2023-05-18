import numpy as np 
import pygame 
import gymnasium as gym 
from gymnasium import spaces 
from collections import deque 




class BallCatch(gym.Env):
    """
    BallCatch game
    """
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, obs_frame=1):
        self.window_size = [600 , 800] ## game window size 
        self.ball_size = 20

        self.ball_start_y = np.round(0.05*(self.window_size[1])) 
        self.ball_middle_y = np.round(0.4*(self.window_size[1]))



        self.bar_length = 150
        self.bar_height = 10 
        self.bar_y_pos = int(np.round(0.9*(self.window_size[1])))
        self.bar_movement_interval = 20  

        self.bar_starting_x_pos = int(np.round(self.window_size[0] * 0.5))

        self.reward_size = 10 


        
        ## for multiple frame as obs 
        self.obs_frame_n = obs_frame
        self.obs_frame_interval = 5
        self.step_count = 1 
        self.max_step = 200
        self.state_n = 5
        self.action_n = 3

        if self.obs_frame_n > 1:
            self.obs_frame_memory = deque(maxlen=100)
            for i in range(1,101):
                self.obs_frame_memory.append(np.zeros(self.state_n))

            self.state_n = obs_frame*self.state_n



        self.observation_space = spaces.Dict({"ball_start": spaces.Box(low=np.array([0,0]), high=np.array(self.window_size), dtype=float),
                                              "ball_middle": spaces.Box(low=np.array([0,0]), high=np.array(self.window_size), dtype=float ),
                                              "bar_location": spaces.Box(low=0 , high=self.window_size[0], dtype=int)
                                              
                                              
                                              })
        
        # three action 0:stop 1:left 2:right 
        self.action_space = spaces.Discrete(3)

        

        self._action_to_direction = { 
            0: np.array([0]),
            1: np.array([-self.bar_movement_interval]),
            2: np.array([self.bar_movement_interval])
        } 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.window = None 
        self.clock = None 

    def _get_obs(self):

        length = np.linalg.norm(self.window_size)
    
        obs = {"ball_start": self._ball_start_location/length,
                    "ball_middle": self._ball_middle_location/length,
                    "bar_location": np.array(self.bar_location/float(self.window_size[0])).reshape([1,])}
        

        obs_array = np.concatenate([obs[key] for key in obs.keys()])
        

        
        if self.obs_frame_n > 1:
            self.obs_frame_memory.append(obs_array) 
            frame_index = [a*(-self.obs_frame_interval)-1 for a in range(self.obs_frame_n)]

            obs_array = np.concatenate([self.obs_frame_memory[i] for i in frame_index])
            print(obs_array)

        return obs_array


    def _get_info(self):
        return None

    def reset(self, seed=None , speed = 10, engergy_transfer_persentage = 1):
        """"
        speed of starting ball can be seleted by user
        energy transfer persentage is how much energy will be transfered to middle ball when hitted 
        """
        
        super().reset(seed=seed)
        self.speed = speed 
        self.step_count = 1 
        self.energy_transfer_persentage = engergy_transfer_persentage


        start_ball_rand_pos = self.np_random.integers(self.ball_size, self.window_size[0] - self.ball_size, size=1,dtype=int)
        middle_ball_rand_pos = self.np_random.integers(self.ball_size, self.window_size[0] - self.ball_size, size=1,dtype=int)
        
        

        self._ball_start_location =  np.array([ int(start_ball_rand_pos) , self.ball_start_y])
        self._ball_middle_location = np.array([ int(middle_ball_rand_pos) , self.ball_middle_y])
        self.bar_location = self.bar_starting_x_pos

        ## movement vector is normalized difference vector between two ball
        self.ball_start_move_vec = self._ball_middle_location - self._ball_start_location
        self.ball_start_move_vec = self.speed*(1/np.linalg.norm(self.ball_start_move_vec))*self.ball_start_move_vec

        self.ball_middle_move_vec = np.array([0,0])
        

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        info = self._get_info()


        return observation, info

    def step(self, action):
        ## map 0:stop , 1:left, 2:right action
        self.step_count += 1 
        direction = self._action_to_direction[action]

        terminated = False 
        reward = 0 

        truncated = True if self.step_count >= self.max_step else False


        ball_hit = self.check_ball_hit() 
        bar_hit = self.check_bar_hit()
        ball_drop = self.check_ball_drop()
        
        if ball_hit:
            self.ball_middle_move_vec = self.ball_start_move_vec*(self.energy_transfer_persentage)**(1/2)
            self.ball_start_move_vec = np.array([0,0])
        
        if bar_hit:
            terminated = True 
            reward = self.reward_size 
        elif ball_drop:
            terminated = True
            difference = abs(float(self.bar_location) - float(self._ball_middle_location[0])) / float(self.window_size[0])
            difference = round(abs(difference),2)
            reward = -self.reward_size*difference
        else:
            self.move_ball()
            self.bar_location = max(self.bar_length/2, min(self.bar_location+direction, int(self.window_size[0]) - self.bar_length/2)) 

            ##difference = abs(float(self.bar_location) - float(self._ball_middle_location[0])) / float(self.window_size[0])
            ##difference = round(abs(difference),2)
            ## reward = -self.reward_size*difference
            reward = 0 

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, truncated, info



    def check_ball_hit(self):

        ball_hit = False

        distance_vector = self._ball_middle_location - self._ball_start_location
        distance = np.linalg.norm(distance_vector)
        if distance <= self.ball_size * 2 :
            ball_hit = True

        if self._ball_middle_location[0] <= 0 or self._ball_middle_location[0] >= self.window_size[0]:
            self.ball_middle_move_vec[0] = -self.ball_middle_move_vec[0]

         

        return ball_hit
    
    def check_bar_hit(self):

        bar_hit = False
        if (
            self.bar_y_pos - self._ball_middle_location[1] < self.ball_size and
            self.bar_location - self.bar_length*(1/2) < self._ball_middle_location[0] and
            self.bar_location + self.bar_length*(1/2) > self._ball_middle_location[0]
        ):
            bar_hit = True 
        return bar_hit
            
    def check_ball_drop(self): 
        
        ball_drop = False
        if self._ball_middle_location[1] > self.bar_y_pos:
            ball_drop = True 
        return ball_drop


    def move_ball(self):
        
        self._ball_start_location = self._ball_start_location + self.ball_start_move_vec
        self._ball_middle_location = self._ball_middle_location + self.ball_middle_move_vec

        
        
    
    def render(self):
        pass

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                self.window_size
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        ## draw the bar(bat)

        pygame.draw.rect(
            canvas,
            (255,0,0),
            pygame.Rect(
            ( int(self.bar_location - 1/2*self.bar_length), self.bar_y_pos ),
            (self.bar_length, self.bar_height)
            ),
        )

        # Now we draw the ball
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._ball_start_location) ,
            self.ball_size,
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._ball_middle_location) ,
            self.ball_size,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

