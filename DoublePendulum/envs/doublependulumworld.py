import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import pygame
import numpy as np

from ..doublependulum.physics.doublependulum import DoublePendulum


class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1/0.03)}
    SCREEN_DIM = 512
    SUCCESS_TIME = 5 # We'll have won if we get the thing to stand up for 5 seconds

    def __init__(self, render_mode=None, size=5, max_F=5, max_ang_vel=4*np.pi, mm=5.0, m1=1.0, l1=1.0, m2=1.0, l2=1.0, dt=0.03, theta_tol=np.pi/10):
        self.size = size  # The size of the track
        self.max_F = max_F # maximum applied F in Neutons
        self.max_ang_vel = max_ang_vel # we'll allow the second pendulum to go faster though 
        self.dt = dt
        if self.dt != 0.03:
            self.metadata = {**self.metadata, "render_fps": int(1 / self.dt)}
        self.max_cart_vel = self.size/self.dt/2 # can we resolve the cart's motion?
        self.theta_tol = theta_tol #absolute tolerance in variance between state and "standing"

        # We don't care about the position of the cart, but we do care about its velocity
        self._target_loc = np.zeros(5) # we don't care about the location of the box
        self._target_loc[0] = np.pi
        self._target_loc[1] = np.pi

        self.mm = mm
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        self.pendulum = DoublePendulum(mm=mm, m1=m1, m2=m2, l1=l1, l2=l2, dt=dt)
        self._agent_location=0.0
        self.state = self._get_obs()

        # State variables...
        # xm - cart position
        # theta1 - angle of  (bounds don't matter, physical setup enforces them)
        # theta2
        # dxm/dt
        # dtheta1/dt
        # dtheta2/dt
        # F_applied
        high = np.array(
            [self.size, 5*np.pi, 5*np.pi, self.max_cart_vel, self.max_ang_vel, 10*(self.m2/self.m1)*self.max_ang_vel, self.max_F], dtype=np.float64
        )
        # Observations are dictionaries with the pendulum's state vector and agent's current applied force
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # We can either add or subtract to the current force by at most 0.5 N on either side
        self.action_space = spaces.Box(-0.5, 0.5, shape=(1,))

        self.success_frames = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

    def _get_obs(self):
        return np.append(self.pendulum.x, self._agent_location)

    def _clip_obs(self, observation):
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the pendulum
        self.pendulum._initialize(seed=seed)

        # Set agent force to 0
        self._agent_location = 0.0

        # Reset the state
        observation = self._get_obs()
        self.state = observation

        # Reset success frames
        self.success_frames = 0

        info = {}

        if self.render_mode == "human":
            self.render()

        return self._clip_obs(observation), info

    def step(self, action):
        # Action will be selected from +/- 0.5
        # Update current force, clipped to current boundaries
        self._agent_location = np.clip(
            self._agent_location + action.squeeze(), -self.max_F, self.max_F
        )
        # Now apply the current force to the pendulum
        self.pendulum.step(F=self._agent_location)

        observation = self._get_obs()
        self.state = observation
        
        # An episode is done iff the pendulum is vertical,
        # and all velocities are < 0.1% of the relevant velocity
        # scale (average pendulum length / dt)
        terminated, offtrack, oob = self._terminate_offtrack_or_oob()
        if terminated:
            reward = 1000
        elif offtrack:
            reward = -100
            terminated = True
        elif oob:
            reward = -10
        else:
            dist = self._distance()
            norm = np.linalg.norm(dist)
            if norm < 1:
                power = 2.0
            else:
                power = 1.0
            reward = -0.1*np.linalg.norm(dist)**power
        
        info = {}

        if self.render_mode == "human":
            self.render()

        return self._clip_obs(observation), reward, terminated, False, info
    
    def _distance(self):
        raw_dist = self.state[1:6] - self._target_loc
        raw_dist[0] = np.atan2(np.sin(raw_dist[0]), np.cos(raw_dist[0]))
        raw_dist[1] = np.atan2(np.sin(raw_dist[1]), np.cos(raw_dist[1]))
        return raw_dist

    def _increment_success_frames_or_reset(self):
        s = self.state
        assert s is not None, "Call reset before using DoublePendulumEnv object."
        thetas = s[1:3]
        cart_coord = self.pendulum.transform_cartesian(do_vels=True)
        vm = cart_coord[3]
        v1 = np.linalg.norm(cart_coord[4])
        v2 = np.linalg.norm(cart_coord[5])
        # we're up if all thetas are pi
        up = np.all([np.abs(np.atan2(np.sin(thet - np.pi), np.cos(thet - np.pi))) < self.theta_tol for thet in thetas])
        # we're "still" if all velocities are less than a tenth%
        # of the relevant length and time scales
        still = np.allclose([vm, v1, v2], 0.0, atol=0.001*np.mean([self.l1, self.l2])/self.dt)
        if up & still:
            self.success_frames += 1
        else:
            self.success_frames = 0
        return s

    def _terminate_offtrack_or_oob(self):
        s = self._increment_success_frames_or_reset()
        terminate = self.success_frames * self.dt >= self.SUCCESS_TIME
        pos = s[0]
        offtrack = np.abs(pos) > self.size
        low = self.observation_space.low
        high = self.observation_space.high
        oob = not (np.all(s >= low) & np.all(s <= high))
        return terminate, offtrack, oob

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state
        cartesian_coords = self.pendulum.transform_cartesian()

        bound = self.l1 + self.l2 + 0.2 + self.size # 7.2 for default
        scale = self.SCREEN_DIM / (bound * 2) # pixels/meter
        offset = self.SCREEN_DIM / 2 # used to transform 0,0 (cartesian) to pix

        if s is None:
            return None

        p0 = [cartesian_coords[0] * scale, 0.0]

        p1 = [
            cartesian_coords[1][0] * scale,
            cartesian_coords[1][1] * scale,
        ]

        p2= [
            cartesian_coords[2][0] * scale,
            cartesian_coords[2][1] * scale,
        ]

        # Our "track"
        pygame.draw.line(
            surf,
            start_pos=(-self.size * scale + offset, 0+offset),
            end_pos=(self.size * scale + offset, 0+offset),
            color=(0, 0, 0),
        )

        # Pendulum links
        pygame.draw.line(
            surf,
            start_pos=(p0[0] + offset, p0[1]+offset),
            end_pos=(p1[0] + offset, p1[1]+offset),
            color=(0, 0, 0),
        )

        pygame.draw.line(
            surf,
            start_pos=(p1[0] + offset, p1[1]+offset),
            end_pos=(p2[0] + offset, p2[1]+offset),
            color=(0, 0, 0),
        )

        # Our cart
        l, r, t, b = int(-0.1*scale), int(0.1*scale), int(0.1*scale), int(-0.1*scale)
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for coord in coords:
            coord = pygame.math.Vector2(coord)
            coord = (coord[0] + p0[0] + offset, coord[1] + p0[1] + offset)
            transformed_coords.append(coord)
        gfxdraw.aapolygon(surf, transformed_coords, (204, 0, 0))
        gfxdraw.filled_polygon(surf, transformed_coords, (0, 0, 0))

        xys = np.array([p1, p2])
        for (x, y) in xys:
            x = x + offset
            y = y + offset
            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
