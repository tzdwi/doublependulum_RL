import math

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import pygame
import numpy as np

from ..physics.doublependulum import DoublePendulum


def _upright_error(theta):
    return math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))


class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1/0.03)}
    SCREEN_DIM = 720
    SUCCESS_TIME = 5 # We'll have won if we get the thing to stand up for 5 seconds

    def __init__(self, render_mode=None, size=50, max_F=25.0, max_ang_vel=4*np.pi, mm=5.0, m1=0.5, l1=10.0, m2=0.5, l2=10.0, dt=0.01, theta_tol=np.pi/10, action_repeat=5):
        self.size = size  # The size of the track
        self.max_F = max_F # maximum applied F in Neutons
        self.max_ang_vel = max_ang_vel # we'll allow the second pendulum to go faster though 
        self.dt = dt
        self.action_repeat = action_repeat # physics steps each chosen force is held for
        self.metadata = {**self.metadata, "render_fps": int(1 / (self.dt * self.action_repeat))}
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
        # Success speed stays 1/3 m/s, the value from a 0.03 s step. It does not follow the integrator step.
        self._still_speed = 0.001 * float(np.mean([self.l1, self.l2])) / 0.03

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
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(8,), dtype=np.float32)

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
        state = np.empty(7, dtype=np.float64)
        state[:6] = self.pendulum.x
        state[6] = self._agent_location
        return state

    def _get_obs(self):
        # Fixed physical scales, so a typical in-range value is about 1.
        # Clipping at +/-5 happens last, and a fall (0.8 rad) arrives before any angle clips.
        state = self._get_state()
        cartesian = self.pendulum.transform_cartesian(do_vels=True)
        obs = np.array(
            [
                state[0] / self.size,
                _upright_error(state[1]) / 0.5,
                _upright_error(state[2]) / 0.5,
                state[3] / 2.0,
                state[4] / 0.5,
                state[5] / 0.5,
                cartesian[4][0] / 1.0,
                cartesian[5][0] / 1.0,
            ],
            dtype=np.float32,
        )
        return self._clip_obs(obs)

    def _clip_obs(self, observation):
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset()
        options = options or {}

        # Odd training episodes start wider and are then shoved. Even episodes and
        # every evaluation reset keep the narrow start. The +20 seed shift preserves parity.
        wide = bool(options.get("training")) and seed is not None and int(seed) % 2 == 1
        self.pendulum._initialize(seed=seed, wide=wide)

        # Optionally make the pendulum perfectly balanced
        # if we're running the demo
        if options.get("balanced"):
            self.pendulum.x[:] = 0
            self.pendulum.x[1] = np.pi
            self.pendulum.x[2] = np.pi
            self.pendulum.x[4] = 0.0
            self.pendulum.x[5] = 0.0
            wide = False

        # Set agent force to 0
        self._agent_location = 0.0
        self.success_frames = 0
        self.state = self._get_state()

        if "push" in options and options["push"] is not None:
            self._shove(float(options["push"]))
        elif wide:
            rng = np.random.default_rng((int(seed), 1))
            self._shove(float(rng.uniform(-self.max_F, self.max_F)))

        observation = self._get_obs()
        self.state = self._get_state()

        info = {}

        if self.render_mode == "human":
            self.render()

        return self._clip_obs(observation), info

    def _shove(self, force):
        # One decision of external force, then clear it. The policy never sees the force, only the motion it leaves.
        self.hold_force(force)
        for _ in range(self.action_repeat):
            info = self._physics_step()
            if info["failure_penalty"]:
                break
        self._agent_location = 0.0
        self.success_frames = 0
        self.state = self._get_state()

    def hold_force(self, action):
        # The action space is a length-1 box, so a vector step sends shape (1,).
        force = float(np.reshape(action, -1)[0])
        self._agent_location = float(np.clip(force, -self.max_F, self.max_F))

    def decision_from_infos(self, infos):
        # Factors are averaged over the physics steps of the decision.
        # A fall ends the episode; the five-second success flag is recorded and does not end it.
        shaping = ["shaped_reward", "upright", "stillness", "position_factor", "control_penalty"]
        events = ["failure_penalty"]
        info = {key: float(np.mean([i[key] for i in infos])) for key in shaping}
        info.update({key: float(sum(i[key] for i in infos)) for key in events})
        info.update({key: any(i[key] for i in infos) for key in ["success", "oob", "offtrack"]})
        terminated = info["failure_penalty"] > 0
        # Once per decision, so bouncing on the rail cannot pile up across physics steps.
        info["offtrack_penalty"] = 0.01 if info["offtrack"] else 0.0
        reward = info["shaped_reward"] - info["offtrack_penalty"]
        return self._clip_obs(self._get_obs()), reward, terminated, info

    def step(self, action):
        # Hold the force for action_repeat physics steps.
        self.hold_force(action)
        infos = []
        for _ in range(self.action_repeat):
            infos.append(self._physics_step())
            if infos[-1]["failure_penalty"]:
                break
        observation, reward, terminated, info = self.decision_from_infos(infos)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _physics_step(self):
        # Apply the current force to the pendulum for one physics step
        self.pendulum.step(F=self._agent_location)
        self.state = self._get_state()
        cartesian = self.pendulum.transform_cartesian(do_vels=True)

        success, offtrack, oob = self._success_offtrack_or_oob(cartesian)
        if offtrack:
            # Clip the pendulum to the track and reverse its momentum.
            self.pendulum.bounce(rail_pos=self.size)
            self.state = self._get_state()
            cartesian = self.pendulum.transform_cartesian(do_vels=True)

        theta1_err = _upright_error(self.state[1])
        theta2_err = _upright_error(self.state[2])
        upright = math.exp(-(theta1_err ** 2 + theta2_err ** 2) / (2 * 0.25 ** 2))
        speed_sq = cartesian[3] ** 2 + math.hypot(*cartesian[4]) ** 2 + math.hypot(*cartesian[5]) ** 2
        stillness = math.exp(-speed_sq / (2 * 0.3 ** 2))
        position_factor = 1.0 - 0.1 * (self.state[0] / self.size) ** 2
        control_penalty = 0.01 * (self.state[6] / self.max_F) ** 2
        shaped_reward = upright * (0.5 + 0.5 * stillness) * position_factor - control_penalty
        failure_penalty = 0.0 if self._head_is_up() else 1.0

        return {
            "shaped_reward": float(shaped_reward),
            "upright": float(upright),
            "stillness": float(stillness),
            "position_factor": float(position_factor),
            "control_penalty": float(control_penalty),
            "failure_penalty": float(failure_penalty),
            "success": bool(success),
            "oob": bool(oob),
            "offtrack": bool(offtrack),
        }
    
    @property
    def healthy(self):
        return self._head_is_up()

    def _head_is_up(self):
        # A fall is either pole more than 0.8 rad from upright.
        limit = 0.8
        return (
            abs(_upright_error(self.state[1])) <= limit
            and abs(_upright_error(self.state[2])) <= limit
        )

    def _distance(self):
        raw_dist = self.state[1:6] - self._target_loc
        raw_dist[0] = np.atan2(np.sin(raw_dist[0]), np.cos(raw_dist[0]))
        raw_dist[1] = np.atan2(np.sin(raw_dist[1]), np.cos(raw_dist[1]))
        return raw_dist

    def _increment_success_frames_or_reset(self, cartesian):
        s = self.state
        assert s is not None, "Call reset before using DoublePendulumEnv object."
        vm = cartesian[3]
        v1 = math.hypot(*cartesian[4])
        v2 = math.hypot(*cartesian[5])
        # we're up if all thetas are pi
        up = (
            abs(_upright_error(s[1])) < self.theta_tol
            and abs(_upright_error(s[2])) < self.theta_tol
        )
        # we're "still" if all velocities are less than a tenth%
        # of the relevant length and time scales
        still = abs(vm) <= self._still_speed and abs(v1) <= self._still_speed and abs(v2) <= self._still_speed
        if up and still:
            self.success_frames += 1
        else:
            self.success_frames = 0
        return s

    def _success_offtrack_or_oob(self, cartesian):
        s = self._increment_success_frames_or_reset(cartesian)
        success = self.success_frames * self.dt >= self.SUCCESS_TIME
        pos = s[0]
        offtrack = abs(pos) > self.size
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

    def render_many(self, plants, alpha=80):
        """Draw the track once and each plant's cart and poles on the same frame.

        Each plant is blended only inside the box it drew in. That matches blending
        the whole frame, because the rest of a plant's layer is blank.
        """
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e
        width = self.SCREEN_DIM
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.Surface((width, width))

        bound = self.l1 + self.l2 + 0.2 + self.size
        scale = width / (bound * 2)
        offset = width / 2

        def draw_body(surf, plant):
            cartesian = plant.pendulum.transform_cartesian()
            p0 = (cartesian[0] * scale + offset, offset)
            p1 = (cartesian[1][0] * scale + offset, cartesian[1][1] * scale + offset)
            p2 = (cartesian[2][0] * scale + offset, cartesian[2][1] * scale + offset)
            cart_size = 0.05 * self.size * scale
            ball_size = max(1, int(0.025 * self.size * scale))
            corners = [
                (p0[0] - cart_size, p0[1] - cart_size),
                (p0[0] - cart_size, p0[1] + cart_size),
                (p0[0] + cart_size, p0[1] + cart_size),
                (p0[0] + cart_size, p0[1] - cart_size),
            ]
            rects = [
                pygame.draw.line(surf, (0, 0, 0), p0, p1),
                pygame.draw.line(surf, (0, 0, 0), p1, p2),
                pygame.draw.polygon(surf, (0, 0, 0), corners),
                pygame.draw.polygon(surf, (204, 0, 0), corners, 1),
            ]
            for point in (p1, p2):
                rects.append(pygame.draw.circle(
                    surf, (204, 204, 0), (int(point[0]), int(point[1])), ball_size))
            return rects[0].unionall(rects[1:]).clip(surf.get_rect())

        base = pygame.Surface((width, width))
        base.fill((255, 255, 255))
        pygame.draw.line(
            base, (0, 0, 0),
            (-self.size * scale + offset, offset),
            (self.size * scale + offset, offset),
        )
        acc = np.flipud(np.transpose(np.array(pygame.surfarray.pixels3d(base)), axes=(1, 0, 2))).astype(np.float32)
        blend = alpha / 255.0
        layer = pygame.Surface((width, width))
        for plant in plants:
            layer.fill((255, 255, 255))
            box = draw_body(layer, plant)
            if box.width == 0 or box.height == 0:
                continue
            # Surface y grows down from the top. The image is flipped, so the crop's rows run the other way.
            image = np.flipud(np.transpose(
                pygame.surfarray.pixels3d(layer)[box.left:box.right, box.top:box.bottom], axes=(1, 0, 2),
            )).astype(np.float32)
            window = acc[width - box.bottom:width - box.top, box.left:box.right]
            mask = np.any(image < 250, axis=-1)
            window[mask] = window[mask] * (1.0 - blend) + image[mask] * blend
        return acc.astype(np.uint8)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
