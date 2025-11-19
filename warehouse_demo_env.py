# Hephaestus-Warehouse-Management-
import time
import numpy as np
import pygame
from pettingzoo.utils.env import ParallelEnv


MOVE = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}
#Pettingzoo, chiedi a chat 
class WarehouseDemo(ParallelEnv):
    metadata = {"name": "WarehouseDemo-v0"}

    def __init__(self, width=12, height=10, n_agents=6, seed=0):
        self.width, self.height = width, height
        self.n_agents = n_agents
        self.possible_agents = [f"robot_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        self.rng = np.random.default_rng(seed)

        # 0=vuoto, 1=scaffale, 2=dock
        self.grid = np.zeros((height, width), dtype=np.int8)
        # qualche scaffale a corridoi
        self.grid[2:8:2, 3] = 1
        self.grid[2:8:2, 8] = 1
        # due dock
        self.grid[height-1, 0] = 2
        self.grid[0, width-1] = 2

        self.max_steps = 400
        self._steps = 0
        self._positions = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self._steps = 0
        empties = list(zip(*np.where(self.grid == 0)))
        self.rng.shuffle(empties)
        self._positions = {a: empties[i] for i, a in enumerate(self.agents)}
        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        self._steps += 1
        rewards = {a: -0.01 for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: self._steps >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}

        desired = {}
        for a in self.agents:
            act = int(actions.get(a, 0))
            dr, dc = MOVE.get(act, (0, 0))
            r, c = self._positions[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] != 1:
                desired[a] = (nr, nc)
            else:
                desired[a] = (r, c)

        # collisions chiedi a chat chatgpt
        counter = {}
        for p in desired.values():
            counter[p] = counter.get(p, 0) + 1

        for a, p in desired.items():
            if counter[p] == 1:
                self._positions[a] = p
            else:
                rewards[a] -= 0.5

            r, c = self._positions[a]
            if self.grid[r, c] == 2:
                rewards[a] += 1.0

        obs = {a: self._observe(a) for a in self.agents}

        if all(trunc.values()):
            self.agents = []
        return obs, rewards, term, trunc, infos

    def _observe(self, agent):
        # (canali) 0=scaffali, 1=dock, 2=agenti
        obs = np.zeros((self.height, self.width, 3), dtype=np.float32)
        obs[:, :, 0] = (self.grid == 1)
        obs[:, :, 1] = (self.grid == 2)
        for _, (r, c) in self._positions.items():
            obs[r, c, 2] = 1.0
        return obs

    # state
    def get_state(self):
        return {
            "grid": self.grid,
            "positions": dict(self._positions),
            "width": self.width,
            "height": self.height,
            "step": self._steps,
            "max_steps": self.max_steps,
            "agents": list(self.agents),
        }

# Pygame (ask to chat gpt)

class PygameRenderer:
    def __init__(self, env: WarehouseDemo, cell=48, margin=2, fps=10):
        pygame.init()
        pygame.display.set_caption("Warehouse MARL - Demo")
        self.env = env
        self.cell = cell
        self.margin = margin
        self.fps = fps
        w = env.width * (cell + margin) + margin
        h = env.height * (cell + margin) + margin + 60  # spazio per HUD
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        # colori
        self.COLOR_BG = (22, 24, 28)
        self.COLOR_EMPTY = (45, 49, 56)
        self.COLOR_SHELF = (120, 120, 120)
        self.COLOR_DOCK = (66, 135, 245)
        self.COLOR_ROBOTS = [
            (239, 83, 80), (102, 187, 106), (255, 202, 40), (126, 87, 194),
            (41, 182, 246), (255, 112, 67), (156, 204, 101), (171, 71, 188)
        ]

    def draw_grid(self):
        self.screen.fill(self.COLOR_BG)
        st = self.env.get_state()
        grid = st["grid"]
        # cells
        for r in range(self.env.height):
            for c in range(self.env.width):
                x = self.margin + c * (self.cell + self.margin)
                y = self.margin + r * (self.cell + self.margin)
                rect = pygame.Rect(x, y, self.cell, self.cell)
                if grid[r, c] == 1:
                    color = self.COLOR_SHELF
                elif grid[r, c] == 2:
                    color = self.COLOR_DOCK
                else:
                    color = self.COLOR_EMPTY
                pygame.draw.rect(self.screen, color, rect, border_radius=8)

        # agents
        for i, (a, (r, c)) in enumerate(st["positions"].items()):
            x = self.margin + c * (self.cell + self.margin) + self.cell // 2
            y = self.margin + r * (self.cell + self.margin) + self.cell // 2
            color = self.COLOR_ROBOTS[i % len(self.COLOR_ROBOTS)]
            pygame.draw.circle(self.screen, color, (x, y), self.cell // 3)

        # HUD
        font = pygame.font.SysFont("consolas", 18)
        text = f"Agents: {len(st['positions'])}   Step: {st['step']}/{st['max_steps']}   FPS: {self.fps}   (R)eset  (ESC) Quit  +/- speed"
        surf = font.render(text, True, (220, 220, 220))
        self.screen.blit(surf, (self.margin, self.env.height * (self.cell + self.margin) + self.margin + 8))

        pygame.display.flip()

    def close(self):
        pygame.quit()

# Demo, still ask to chatgpt

if __name__ == "__main__":
    env = WarehouseDemo(width=12, height=10, n_agents=6, seed=42)
    obs, infos = env.reset()
    renderer = PygameRenderer(env, cell=40, margin=3, fps=8)

    running = True
    while running:
        # eventi tastiera / finestra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_r, pygame.K_R):
                    obs, infos = env.reset()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    renderer.fps = min(60, renderer.fps + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    renderer.fps = max(1, renderer.fps - 1)

        # azioni random (sostituisci con la tua policy/algoritmo)
        actions = {a: np.random.randint(0, 5) for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)

        renderer.draw_grid()
        renderer.clock.tick(renderer.fps)

        # ricomincia episodio quando finisce il tempo
        if not env.agents:
            obs, infos = env.reset()

    renderer.close()
