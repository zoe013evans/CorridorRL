from gym import Env
from gym import spaces
import numpy as np 
import random 
import pygame 


class guideRobot(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=13, human_jumps=1):
        self.size = size 
        self.window_size = 700
        self.training_mode = True

        #the observation space: 

        self.observation_space = spaces.Dict(
            {
            "agent": spaces.Box(0, np.array([size-1,0]), dtype=int),
            "target": spaces.Box(0, np.array([size-1,0]), dtype=int),
            "human": spaces.Box(0, np.array([size-1,0]), dtype=int),
        
           }
        )

        self.action_space = spaces.Discrete(6)

        self._action_to_direction = {
        
            0: np.array([1,0]),
            1: np.array([-1,0]),
            2: np.array([2,0]),
            3: np.array([-2,0]),
            4: np.array([3,0]),
            5: np.array([-3,0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._jump_size = human_jumps




    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "human": self._human_location}
    
    def _get_info(self): 
        return{"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def step(self, action):

        #Setting current location to last location 
        self._agent_last_location = self._agent_location

        #Converting the action to a numerical value to add:
        direction = self._action_to_direction[action]

        #Setting the agents new location, a clipped version of the location + direction 
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size-1)



        # MOVING THE HUMAN: 
        [a,b] = self._agent_location 
        [c,d] = self._human_location


        human_robot_dist = np.linalg.norm(self._agent_location - self._human_location, ord=1)
        if human_robot_dist > 2:
            self._human_location = self._human_location
        else: 
            if a > c: 
                #self._human_location = self._human_location + [self._jump_size,0]
                self._human_location = np.clip(self._human_location + [self._jump_size,0], 0, self.size-1)

            elif a < c: 
                self._human_location = self._human_location + [-self._jump_size,0]
            else: 
                self._human_location = self._human_location + [0,0] 

        #TERMINATION? 
        
        terminated = np.array_equal(self._human_location, self._target_location)
        if terminated: 
            reward = 1 
        else:
            if human_robot_dist > 2:
                reward = -1
            else: 
                reward = -1 

        observation = self._get_obs()
        info = self._get_info

        if self.render_mode == "human":
            self._render_frame()


        return observation, reward, terminated, False, info


     
        
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frane()



    def _render_frame(self):


        pix_square_size = (self.window_size/self.size)

        if self.window is None and self.render_mode == "human": 
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode((self.window_size, pix_square_size))
        
        if self.clock is None and self.render_mode == "human": 
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, pix_square_size))
        canvas.fill((255,255,255))
        #pix_square_size = (self.window_size/self.size)


        #Drawing the goal:

        pygame.draw.rect(
            canvas, 
            (255,0,0), 
            pygame.Rect(
                pix_square_size*self._target_location, (pix_square_size,pix_square_size)
            )
        )

        #Drawing agent:

        pygame.draw.circle(
            canvas, 
            (0,0,255),
            (self._agent_location + 0.5) * pix_square_size, pix_square_size/3
        )

        #Drawing human: 

        pygame.draw.circle(
            canvas, 
            (0,200,100),
            (self._human_location + 0.5) * pix_square_size, pix_square_size/3
        )

        #Drawing Lines: 
        
        for x in range(self.size + 1): 
            
            pygame.draw.line(
                canvas, 
                0, 
                (pix_square_size * x,0),
                (pix_square_size *x, self.window_size), 
                width=3
            )

        if self.render_mode == "human": 
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else: 
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))



    def reset(self, options=None):
        super().reset()
        

        #Agent and Human location! 
        self._agent_last_location = [1,0]

    
        #When training: 

        if self.training_mode == True: 
            
            n = random.randint(0,self.size-1)
            m = random.randint(0,self.size-1)
            
            self._human_location = np.array([n,0])
            self._agent_location = np.array([0,0])
            #print("training: ", self._agent_location, self._human_location)
        else: 
            #When testing:
            #print("testing mode")
            self._agent_location = np.array([1,0])
            self._human_location = np.array([0,0])

        

        #self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        

        self._target_location = np.array([self.size-1,0])
        
        
        observation = self._get_obs()
        info = self._get_info()

        

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def close(self):
        pass









