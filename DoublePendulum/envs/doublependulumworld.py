import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import pygame
import numpy as np

from ..physics.doublependulum import DoublePendulum


class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1/0.03)}
    SCREEN_DIM = 720
    SUCCESS_TIME = 5 # We'll have won if we get the thing to stand up for 5 seconds

    def __init__(self, render_mode=None, size=50, max_F=25.0, max_ang_vel=4*np.pi, mm=5.0, m1=0.5, l1=10.0, m2=0.5, l2=10.0, dt=0.03, theta_tol=np.pi/10):
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

        # at current defaults, speed scale is ~3.333 m/s
        self.speed_scale = 0.01*np.mean([self.l1, self.l2])/self.dt

        self.pendulum = DoublePendulum(mm=mm, m1=m1, m2=m2, l1=l1, l2=l2, dt=dt)
        self._agent_location=0.0
        self.state = self._get_state()

        # State variables...
        # xm - cart position
        # theta1 - angle of  (bounds don't matter, physical setup enforces them)
        # theta2
        # dxm/dt
        # dtheta1/dt
        # dtheta2/dt
        # F_applied
        self._state_high = np.array(
            [self.size, 5*np.pi, 5*np.pi, self.max_cart_vel, self.max_ang_vel, 10*(self.m2/self.m1)*self.max_ang_vel, self.max_F], dtype=np.float64
        )
        self._state_low = -self._state_high
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        # action is the applied force over that timestep
        self.action_space = spaces.Box(-max_F, max_F, shape=(1,))

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

    def _get_state(self):
        # This is the raw physical state
        return np.append(self.pendulum.x, self._agent_location)

    def _get_obs(self):
        # now lets scale our raw state into a helpful observation
        # that is bounded in all dimensions to -1, 1
        # cart pos scaled to rail size
        # thetas wrapped around pi, scaled by pi/4
        # cart velocity relative to the speed scale defined by the physics
        # angular velocities scaled by pi
        # foce relative to the max
        state = self._get_state()
        obs = np.array(
            [
                state[0] / self.size,
                np.arctan2(np.sin(state[1] - np.pi), np.cos(state[1] - np.pi)) / (np.pi / 4),
                np.arctan2(np.sin(state[2] - np.pi), np.cos(state[2] - np.pi)) / (np.pi / 4),
                state[3] / self.speed_scale,
                state[4] / np.pi,
                state[5] / np.pi,
                state[6] / self.max_F,
            ],
            dtype=np.float32,
        )
        return self._clip_obs(obs)

    def _clip_obs(self, observation):
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def reset(self, seed=42, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the pendulum
        self.pendulum._initialize(seed=seed)

        # Optionally make the pendulum perfectly balanced
        # if we're running the demo
        if options and options.get("balanced"):
            self.pendulum.x[:] = 0
            self.pendulum.x[1] = np.pi
            self.pendulum.x[2] = np.pi
            self.pendulum.x[4] = 0.0
            self.pendulum.x[5] = 0.0

        # Set agent force to 0
        self._agent_location = 0.0

        # Reset the state
        observation = self._get_obs()
        self.state = self._get_state()

        # Reset success frames
        self.success_frames = 0

        info = {}

        if self.render_mode == "human":
            self.render()

        return self._clip_obs(observation), info

    def step(self, action):
        # Update current force, clipped to current boundaries
        self._agent_location = np.clip(
            action, -self.max_F, self.max_F
        )
        # Now apply the current force to the pendulum
        self.pendulum.step(F=self._agent_location)

        observation = self._get_obs()
        self.state = self._get_state()

        terminated = False
        success, offtrack, oob = self._success_offtrack_or_oob()
        if offtrack:
            # Clip the pendulum to the track and reverse its momentum.
            self.pendulum.bounce(rail_pos=self.size)
            observation = self._get_obs()
            self.state = self._get_state()

        length = self.l1 + self.l2
        cart_coord = self.pendulum.transform_cartesian(do_vels=True)
        tip_x = cart_coord[2][0] / length
        tip_y = cart_coord[2][1] / length
        distance_penalty = (1.0 - tip_y) ** 2.0 + 0.01 * tip_x ** 2.0
        velocity_penalty = (
            0.01 * (self.state[3] / self.speed_scale) ** 2.0
            + 0.01 * (self.state[4] / self.max_ang_vel) ** 2.0
            + 0.01 * (self.state[5] / self.max_ang_vel) ** 2.0
        )
        theta1_err = np.arctan2(np.sin(self.state[1] - np.pi), np.cos(self.state[1] - np.pi))
        theta2_err = np.arctan2(np.sin(self.state[2] - np.pi), np.cos(self.state[2] - np.pi))
        tilt_penalty = 0.1 * min(1.0, (theta1_err**2 + theta2_err**2) / (2 * self.theta_tol**2))
        control_penalty = 0.001 * (self.state[6] / self.max_F) ** 2.0
        # Must be at least the capped tilt penalty / (1 - gamma), so falling is never the cheaper option.
        failure_penalty = 0.0 if success or self.healthy else 10.0
        offtrack_penalty = 1.0 if offtrack else 0.0
        success_bonus = 10.0 if success else 0.0
        if success or failure_penalty:
            terminated = True

        info = {
            "distance_penalty": float(distance_penalty),
            "velocity_penalty": float(velocity_penalty),
            "tilt_penalty": float(tilt_penalty),
            "control_penalty": float(control_penalty),
            "failure_penalty": float(failure_penalty),
            "offtrack_penalty": float(offtrack_penalty),
            "success_bonus": float(success_bonus),
            "success": bool(success),
            "oob": bool(oob),
            "offtrack": bool(offtrack),
        }
        reward = (
            -info["distance_penalty"]
            -info["velocity_penalty"]
            -info["tilt_penalty"]
            -info["control_penalty"]
            -info["failure_penalty"]
            -info["offtrack_penalty"]
            +info["success_bonus"]
        )

        if self.render_mode == "human":
            self.render()

        return self._clip_obs(observation), reward, terminated, False, info
    
    @property
    def healthy(self):
        # We're healthy if the second head is above the first
        # and both are above 0.5 * l1
        cart_coord = self.pendulum.transform_cartesian(do_vels=True)
        y2 = cart_coord[2][1]/(self.l1+self.l2)/0.3 # originally, each pole was 0.6
        return y2 >= 1

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

    def _success_offtrack_or_oob(self):
        s = self._increment_success_frames_or_reset()
        success = self.success_frames * self.dt >= self.SUCCESS_TIME
        pos = s[0]
        offtrack = np.abs(pos) > self.size
        oob = not (np.all(s >= self._state_low) & np.all(s <= self._state_high))
        return success, offtrack, oob

    def render(self, log_reward=False, reward=None):
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

        bound = self.l1 + self.l2 + 0.2 + self.size # 52.2 for default
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

        # Our cart is 5% of the size of the track
        cart_size = 0.05*self.size*scale
        ball_size = 0.025*self.size*scale
        l, r, t, b = int(-cart_size), int(cart_size), int(cart_size), int(-cart_size)
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
            gfxdraw.aacircle(surf, int(x), int(y), int(ball_size), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(ball_size), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if log_reward and reward is not None:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.Font(None, 24)
            self.screen.blit(font.render(f"Reward = {reward:.3f}", True, (0, 0, 0)), (10, 10))

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
