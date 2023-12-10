import numpy as np
from physics_sim import PhysicsSim
import itertools

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 9
        self.action_low = 10
        self.action_high = 900
        self.action_size = 4
        self.action_split = 15 # 3 SPEED LEVELS 400,450,500

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.max_height_reward = 5       #5 is max reward after takeoff upon reaching max desired height
        self.max_height = 20
        self.max_height_acheived = False
        self.max_height_target = np.array([0., 0., 30.])
        self.target_velocity = 0.0 #  np.array([0., 0., 0.])
        self.Has_landed = False
        
        self.rotor_speed_levels =  np.round(np.linspace(self.action_low,self.action_high,self.action_split),0)

    def get_reward(self,done):
        """Uses current pose of sim to return reward."""
#         reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        if done:
            if self.sim.time < 10.:
                reward = -100
            else:
                reward = 0
        else:
            reward = self.calculate_reward(self.sim.pose)
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        
        rotor_speeds = self.eval_rotor_speed(rotor_speeds)
        
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
            pose_all.append(np.concatenate([self.sim.pose,self.sim.v,[self.max_height_acheived]]))
            
        if done or self.Has_landed:
            next_state = self.reset()
            done = True
        else:
            next_state = np.concatenate(pose_all)
            
        #print('next state',next_state,'reward',reward)
        return next_state, reward, done
    
    def eval_rotor_speed(self,rotor_speeds):

        calc_rotor_speed = [self.rotor_speed_levels[idx] for idx in rotor_speeds]
        return calc_rotor_speed

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([np.concatenate([self.sim.pose,self.sim.v,[self.max_height_acheived]])] * self.action_repeat)
        self.max_height_acheived = False
        self.Has_landed = False
        return state
    
    def calculate_reward(self, pose):

        z = pose[2]

        if z <= self.max_height and not self.max_height_acheived:
            
            reward = 10./np.exp(abs(self.sim.pose[2] - self.max_height)) 
            
        else:
            
            diff = 0.1 *(abs(self.sim.pose[:3] - self.target_pos).sum())
            
            reward = 100./np.exp(diff)
            
            if self.sim.pose[2] <= 19:
                reward = reward + 100/np.exp(0.01*abs(self.sim.v[2] - self.target_velocity))
            
            self.max_height_acheived = True
        
            if (np.round(np.array(self.sim.pose),0)==np.array(self.target_pos)):
                reward = reward+ 1000
                self.Has_landed = True
                print('Egle Has landed...')

            
        return reward
    
    