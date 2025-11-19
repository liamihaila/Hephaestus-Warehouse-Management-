# Hephaestus-Warehouse-Management (Tasks + MARL-ready)
# Compatible with PettingZoo ParallelEnv API
# - Two docks: INBOUND (new items) and OUTBOUND (packaging)
# - Tasks:
#     * inbound: pick at INBOUND dock, drop at a shelf slot
#     * outbound: pick at a shelf slot, drop at OUTBOUND dock
# - Cooperative reward, with catastrophic penalty (-inf) on collisions/bump/swap
# - Action set: 0..4 = move, 5 = PICKUP_ITEM, 6 = DROP_ITEM
# - Observations: channels (shelves, inbound dock, outbound dock, items on floor, agents, agent-cargo)
# - Includes a minimal pygame renderer for visualization

import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

MOVE = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}
PICKUP = 5
DROP = 6

INBOUND = 2  # new-item dock
OUTBOUND = 3 # packaging dock
SHELF = 1
EMPTY = 0

class WarehouseTasksEnv(ParallelEnv):
    metadata = {"name": "WarehouseTasks-v0"}

    def __init__(
        self,
        width: int = 12,
        height: int = 10,
        n_agents: int = 6,
        seed: int = 0,
        max_steps: int = 400,
        n_inbound_tasks: int = 6,
        n_outbound_tasks: int = 6,
        coop_rewards: bool = True,
        pickup_reward: float = 1.0,
        deliver_reward: float = 5.0,
        stay_penalty: float = 0.01,
    ):
        """Cooperative warehouse tasks environment.
        Args set core grid and reward shaping. Collisions cause -inf team reward.
        """
        self.width, self.height = width, height
        self.n_agents = n_agents
        self.possible_agents = [f"robot_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        self.max_steps = max_steps
        self._steps = 0
        self.rng = np.random.default_rng(seed)

        # Reward config
        self.coop_rewards = coop_rewards
        self.pickup_reward = pickup_reward
        self.deliver_reward = deliver_reward
        self.stay_penalty = stay_penalty

        # Grid: 0 empty, 1 shelf (blocking), 2 inbound dock, 3 outbound dock
        self.grid = np.zeros((height, width), dtype=np.int8)
        # Shelves as blocking columns creating corridors
        self.grid[2:8:2, 3] = SHELF
        self.grid[2:8:2, 8] = SHELF
        # Docks
        self.inbound_pos = (height - 1, 0)
        self.outbound_pos = (0, width - 1)
        self.grid[self.inbound_pos] = INBOUND
        self.grid[self.outbound_pos] = OUTBOUND

        # Task config
        self.n_inbound_tasks = n_inbound_tasks
        self.n_outbound_tasks = n_outbound_tasks

        # Runtime state
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._cargo: Dict[str, Optional[int]] = {}  # task_id carried or None
        self._task_src: Dict[int, Tuple[int, int]] = {}  # task_id -> source pos
        self._task_dst: Dict[int, Tuple[int, int]] = {}  # task_id -> dest pos
        self._task_active: Dict[int, bool] = {}          # picked but not delivered
        self._items_on_floor: Dict[Tuple[int, int], int] = {}  # pos -> task_id (outbound src only)
        self._inbound_queue: List[int] = []  # list of inbound task_ids waiting at inbound dock
        self._delivered: int = 0

        # Precompute candidate shelf slots (walkable cells adjacent to a shelf)
        self.shelf_slots = self._compute_shelf_slots()

        # Gym spaces
        n_actions = 7
        H, W = self.height, self.width
        # channels: 0 shelves, 1 inbound, 2 outbound, 3 items, 4 agents, 5 my cargo
        self._obs_shape = (H, W, 6)
        self._action_spaces = {a: Discrete(n_actions) for a in self.possible_agents}
        self._observation_spaces = {a: Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=np.float32) for a in self.possible_agents}

    # ---- PettingZoo spaces API ----
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ---- Reset & step ----
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self._steps = 0
        self._delivered = 0

        # place agents uniformly at random on empty cells (not shelves or docks)
        empties = list(zip(*np.where(self.grid == EMPTY)))
        empties = [p for p in empties if p not in (self.inbound_pos, self.outbound_pos)]
        self.rng.shuffle(empties)
        if len(empties) < len(self.agents):
            raise RuntimeError("Not enough empty cells to place agents.")
        self._positions = {a: empties[i] for i, a in enumerate(self.agents)}
        self._cargo = {a: None for a in self.agents}

        # Build task set
        self._task_src.clear(); self._task_dst.clear(); self._task_active.clear()
        self._items_on_floor.clear(); self._inbound_queue.clear()

        # Inbound tasks: src=inbound dock, dst=random shelf slot
        for k in range(self.n_inbound_tasks):
            tid = k
            self._task_src[tid] = self.inbound_pos
            self._task_dst[tid] = self.rng.choice(self.shelf_slots)
            self._task_active[tid] = False
            self._inbound_queue.append(tid)

        # Outbound tasks: src=random shelf slot, dst=outbound dock
        base = self.n_inbound_tasks
        # choose distinct shelf slots for outbound items
        slots = self.rng.choice(self.shelf_slots, size=min(self.n_outbound_tasks, len(self.shelf_slots)), replace=False)
        for j, slot in enumerate(slots):
            tid = base + j
            self._task_src[tid] = tuple(slot)
            self._task_dst[tid] = self.outbound_pos
            self._task_active[tid] = False
            self._items_on_floor[tuple(slot)] = tid

        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        self._steps += 1

        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: self._steps >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Apply slight penalty for staying still
        for a, act in actions.items():
            if act == 0:
                rewards[a] -= self.stay_penalty

        # First compute desired next positions (without resolving conflicts)
        desired: Dict[str, Tuple[int, int]] = {}
        for a in self.agents:
            act = int(actions.get(a, 0))
            r, c = self._positions[a]
            if act in MOVE:
                dr, dc = MOVE[act]
                nr, nc = r + dr, c + dc
                if self._is_walkable(nr, nc):
                    desired[a] = (nr, nc)
                else:
                    desired[a] = (r, c)
            else:
                desired[a] = (r, c)

        # Detect conflicts (same target) and swaps
        collisions = set()  # agents that collided
        # same target
        counter = {}
        for p in desired.values():
            counter[p] = counter.get(p, 0) + 1
        for a, p in desired.items():
            if counter[p] > 1 and p != self._positions[a]:
                collisions.add(a)
        # swaps: a->b_pos and b->a_pos
        pos2agent = {pos: ag for ag, pos in self._positions.items()}
        for a, p_a in desired.items():
            pos_a = self._positions[a]
            if p_a in pos2agent:
                b = pos2agent[p_a]
                if b != a:
                    p_b = desired.get(b, self._positions[b])
                    if p_b == pos_a and p_a != pos_a:
                        collisions.add(a)
                        collisions.add(b)
        # moving into someone who stays (counts above because counter>1), but ensure both flagged
        for a, p in desired.items():
            if p in self._positions.values() and p != self._positions[a]:
                # someone currently there; if they also desire staying there -> bump
                occ_agent = pos2agent[p]
                if desired.get(occ_agent, self._positions[occ_agent]) == p:
                    collisions.add(a)
                    collisions.add(occ_agent)

        # Resolve movement for non-colliding agents
        for a in self.agents:
            if a not in collisions:
                self._positions[a] = desired[a]

        # Handle PICKUP/DROP (after movement)
        for a in self.agents:
            act = int(actions.get(a, 0))
            if a in collisions:
                continue
            if act == PICKUP:
                rewards[a] += self._try_pickup(a)
            elif act == DROP:
                gained, delivered = self._try_drop(a)
                rewards[a] += gained
                self._delivered += delivered

        # Catastrophic collision penalty: -inf (team-wide, cooperative catastrophe)
        any_collision = len(collisions) > 0
        if any_collision:
            for a in self.agents:
                rewards[a] = -np.inf

        # Cooperative aggregation (if enabled)
        if self.coop_rewards:
            team_reward = sum(rewards.values())
            # If any -inf present, team_reward becomes -inf. Broadcast.
            if any(np.isneginf(r) for r in rewards.values()):
                team_reward = -np.inf
            rewards = {a: team_reward for a in self.agents}

        # Termination when all tasks done
        all_tasks = len(self._task_src)
        done_all = (self._delivered >= all_tasks)
        for a in self.agents:
            terminated[a] = done_all

        obs = {a: self._observe(a) for a in self.agents}

        # End episode if everyone terminated or all truncated (PZ parallel spec)
        if all(terminated.values()) or all(truncated.values()):
            self.agents = []
        return obs, rewards, terminated, truncated, infos

    # ---- Helpers ----
    def _is_walkable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        # shelves are blocking
        if self.grid[r, c] == SHELF:
            return False
        return True

    def _compute_shelf_slots(self) -> List[Tuple[int, int]]:
        slots = []
        H, W = self.height, self.width
        for r in range(H):
            for c in range(W):
                if self.grid[r, c] == SHELF:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and self.grid[nr, nc] == EMPTY:
                            slots.append((nr, nc))
        # unique
        slots = sorted(list(set(slots)))
        return slots

    def _try_pickup(self, agent: str) -> float:
        if self._cargo[agent] is not None:
            return 0.0  # already carrying
        pos = self._positions[agent]
        # outbound item on floor?
        if pos in self._items_on_floor:
            tid = self._items_on_floor.pop(pos)
            self._cargo[agent] = tid
            self._task_active[tid] = True
            return self.pickup_reward
        # inbound queue at inbound dock?
        if pos == self.inbound_pos and len(self._inbound_queue) > 0:
            tid = self._inbound_queue.pop(0)
            self._cargo[agent] = tid
            self._task_active[tid] = True
            return self.pickup_reward
        return 0.0

    def _try_drop(self, agent: str) -> Tuple[float, int]:
        # returns (reward_delta, delivered_flag 0/1)
        tid = self._cargo.get(agent, None)
        if tid is None:
            return 0.0, 0
        pos = self._positions[agent]
        dst = self._task_dst[tid]
        if pos == dst:
            # Successful delivery
            self._cargo[agent] = None
            self._task_active[tid] = False
            return self.deliver_reward, 1
        else:
            return 0.0, 0

    def _observe(self, agent: str) -> np.ndarray:
        obs = np.zeros(self._obs_shape, dtype=np.float32)
        obs[:, :, 0] = (self.grid == SHELF)
        obs[:, :, 1] = (self.grid == INBOUND)
        obs[:, :, 2] = (self.grid == OUTBOUND)
        # items on floor (outbound sources)
        for (r, c), _ in self._items_on_floor.items():
            obs[r, c, 3] = 1.0
        # agents
        for _, (r, c) in self._positions.items():
            obs[r, c, 4] = 1.0
        # self cargo bit (broadcast at my position for convenience)
        if self._cargo[agent] is not None:
            r, c = self._positions[agent]
            obs[r, c, 5] = 1.0
        return obs

    # Expose compact global state (for debugging or centralized critics)
    def get_state(self):
        return {
            "grid": self.grid.copy(),
            "positions": dict(self._positions),
            "cargo": dict(self._cargo),
            "items_on_floor": dict(self._items_on_floor),
            "inbound_queue_len": len(self._inbound_queue),
            "delivered": self._delivered,
            "steps": self._steps,
            "max_steps": self.max_steps,
            "agents": list(self.agents),
        }

# ---------------- Pygame renderer for visualization -----------------
class PygameRenderer:
    def __init__(self, env: WarehouseTasksEnv, cell=48, margin=2, fps=10):
        pygame.init()
        pygame.display.set_caption("Warehouse MARL - Tasks Demo")
        self.env = env
        self.cell = cell
        self.margin = margin
        self.fps = fps
        w = env.width * (cell + margin) + margin
        h = env.height * (cell + margin) + margin + 64
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        # colors
        self.COLOR_BG = (22, 24, 28)
        self.COLOR_EMPTY = (45, 49, 56)
        self.COLOR_SHELF = (120, 120, 120)
        self.COLOR_IN = (66, 135, 245)
        self.COLOR_OUT = (255, 170, 35)
        self.COLOR_ITEM = (200, 200, 50)
        self.COLOR_ROBOTS = [
            (239, 83, 80), (102, 187, 106), (255, 202, 40), (126, 87, 194),
            (41, 182, 246), (255, 112, 67), (156, 204, 101), (171, 71, 188)
        ]

    def draw(self):
        self.screen.fill(self.COLOR_BG)
        st = self.env.get_state()
        grid = st["grid"]

        # cells
        for r in range(self.env.height):
            for c in range(self.env.width):
                x = self.margin + c * (self.cell + self.margin)
                y = self.margin + r * (self.cell + self.margin)
                rect = pygame.Rect(x, y, self.cell, self.cell)
                val = grid[r, c]
                if val == SHELF:
                    color = self.COLOR_SHELF
                elif val == INBOUND:
                    color = self.COLOR_IN
                elif val == OUTBOUND:
                    color = self.COLOR_OUT
                else:
                    color = self.COLOR_EMPTY
                pygame.draw.rect(self.screen, color, rect, border_radius=8)

        # items on floor
        for (r, c), _ in st["items_on_floor"].items():
            x = self.margin + c * (self.cell + self.margin)
            y = self.margin + r * (self.cell + self.margin)
            rect = pygame.Rect(x + self.cell//4, y + self.cell//4, self.cell//2, self.cell//2)
            pygame.draw.rect(self.screen, self.COLOR_ITEM, rect, border_radius=6)

        # agents
        for i, (a, (r, c)) in enumerate(st["positions"].items()):
            cx = self.margin + c * (self.cell + self.margin) + self.cell // 2
            cy = self.margin + r * (self.cell + self.margin) + self.cell // 2
            color = self.COLOR_ROBOTS[i % len(self.COLOR_ROBOTS)]
            pygame.draw.circle(self.screen, color, (cx, cy), self.cell // 3)
            # cargo ring
            if st["cargo"][a] is not None:
                pygame.draw.circle(self.screen, (240, 240, 240), (cx, cy), self.cell // 3, width=3)

        # HUD
        font = pygame.font.SysFont("consolas", 18)
        text = (
            f"Agents: {len(st['positions'])}   "
            f"Delivered: {st['delivered']}/{len(self.env._task_src)}   "
            f"InboundQ: {st['inbound_queue_len']}   "
            f"Step: {st['steps']}/{st['max_steps']}   "
            f"FPS: {self.fps}   (R)eset  (ESC) Quit  +/- speed"
        )
        surf = font.render(text, True, (220, 220, 220))
        self.screen.blit(surf, (self.margin, self.env.height * (self.cell + self.margin) + self.margin + 8))

        pygame.display.flip()

    def close(self):
        pygame.quit()

# ----------------- Manual demo loop ------------------
if __name__ == "__main__":
    env = WarehouseTasksEnv(width=12, height=10, n_agents=6, seed=42,
                            n_inbound_tasks=6, n_outbound_tasks=6,
                            coop_rewards=True)
    obs, infos = env.reset()
    renderer = PygameRenderer(env, cell=40, margin=3, fps=8)

    running = True
    while running:
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

        # Random policy for demo (replace with your MARL algorithm)
        actions = {}
        for a in env.agents:
            # 80% move randomly, 10% pickup, 10% drop
            u = env.rng.random()
            if u < 0.1:
                actions[a] = PICKUP
            elif u < 0.2:
                actions[a] = DROP
            else:
                actions[a] = env.rng.integers(0, 5)
        obs, rewards, term, trunc, infos = env.step(actions)

        renderer.draw()
        renderer.clock.tick(renderer.fps)

        if not env.agents:  # episode ended
            obs, infos = env.reset()

    renderer.close()
