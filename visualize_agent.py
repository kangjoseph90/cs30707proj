"""Visualize trained agents playing Snake with pygame.

Usage:
    python visualize_agent.py --agent ego_merged_obst_k29
    python visualize_agent.py --agent all --speed 10
    python visualize_agent.py --agent blind_sniff --episodes 5

Controls:
    SPACE  - pause/resume
    UP/DOWN - speed up / slow down
    ESC    - quit
    N      - next episode
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame
import torch

from snake_env import SnakeEnv
from train_snake import make_encoder
from model import MLPDQN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 15
PPB = 32  # pixels per block
MARGIN = 4  # gap between agents
HEADER_H = 48
FPS = 60

# Colors
COL_BG = (30, 30, 30)
COL_GRID = (50, 50, 50)
COL_EMPTY = (40, 40, 40)
COL_WALL = (100, 100, 100)
COL_BODY = (0, 200, 100)
COL_HEAD = (0, 255, 150)
COL_FRUIT = (255, 60, 60)
COL_TEXT = (220, 220, 220)
COL_SCORE = (255, 255, 100)
COL_DEAD = (180, 40, 40)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

AGENTS = {
    "ego_merged_obst_k29": {
        "rep": "egocentric_merged_obstacle_local",
        "ws": 29,
        "ckpt": "results/sweep_500k_final/egocentric_merged_obstacle_local_w29_seed0/checkpoint.pt",
        "label": "Ego Obst K=29",
    },
    "ego_merged_ord_k29": {
        "rep": "egocentric_merged_ordered_local",
        "ws": 29,
        "ckpt": "results/sweep_500k_final/egocentric_merged_ordered_local_w29_seed0/checkpoint.pt",
        "label": "Ego Ord K=29",
    },
    "ego_merged_obst_k5": {
        "rep": "egocentric_merged_obstacle_local",
        "ws": 5,
        "ckpt": "results/sweep_500k_final/egocentric_merged_obstacle_local_w5_seed0/checkpoint.pt",
        "label": "Ego Obst K=5",
    },
    "ego_merged_ord_k5": {
        "rep": "egocentric_merged_ordered_local",
        "ws": 5,
        "ckpt": "results/sweep_500k_final/egocentric_merged_ordered_local_w5_seed0/checkpoint.pt",
        "label": "Ego Ord K=5",
    },
    "merged_obst_k5": {
        "rep": "merged_obstacle_local",
        "ws": 5,
        "ckpt": "results/sweep_500k_final/merged_obstacle_local_w5_seed0/checkpoint.pt",
        "label": "Merged Obst K=5",
    },
    "merged_obst_k29": {
        "rep": "merged_obstacle_local",
        "ws": 29,
        "ckpt": "results/sweep_500k_final/merged_obstacle_local_w29_seed0/checkpoint.pt",
        "label": "Merged Obst K=29",
    },
    "blind_sniff": {
        "rep": "blind_sniff",
        "ws": None,
        "ckpt": "results/sweep_500k_final/blind_sniff_seed0/checkpoint.pt",
        "label": "Blind Sniff",
    },
    "ego_merged_obst_k9": {
        "rep": "egocentric_merged_obstacle_local",
        "ws": 9,
        "ckpt": "results/sweep_500k_final/egocentric_merged_obstacle_local_w9_seed0/checkpoint.pt",
        "label": "Ego Obst K=9",
    },
    "heuristic": {
        "rep": None,
        "ws": None,
        "ckpt": None,
        "label": "Heuristic",
    },
}


def heuristic_action(env, state):
    """Simple heuristic: avoid death + approach fruit."""
    from snake_env import DELTA, get_relative_dirs
    bs = env.board_size
    hx, hy = state.head
    fx, fy = state.fruit
    body_set = set(state.body)
    abs_dirs = get_relative_dirs(state.direction)
    best_action = 1
    best_dist = 999
    any_ok = False
    for i, ad in enumerate(abs_dirs):
        dx, dy = DELTA[ad]
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= bs or ny < 0 or ny >= bs:
            continue
        if (nx, ny) in body_set:
            if (nx, ny) == state.body[-1] and (nx, ny) != (fx, fy):
                pass
            else:
                continue
        any_ok = True
        dist = abs(nx - fx) + abs(ny - fy)
        if dist < best_dist:
            best_dist = dist
            best_action = i
    if not any_ok:
        return 1
    return best_action


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_board(surface, state, x_off, y_off, alive, board_size, ppb):
    """Draw one Snake game board at the given offset."""
    hx, hy = state.head
    body_set = set(state.body)

    # Grid background
    for by in range(board_size):
        for bx in range(board_size):
            rect = pygame.Rect(x_off + bx * ppb, y_off + by * ppb, ppb, ppb)
            pygame.draw.rect(surface, COL_EMPTY, rect)
            pygame.draw.rect(surface, COL_GRID, rect, 1)

    # Body
    for seg in state.body:
        sx, sy = seg
        rect = pygame.Rect(x_off + sx * ppb + 1, y_off + sy * ppb + 1, ppb - 2, ppb - 2)
        color = COL_DEAD if not alive else COL_BODY
        pygame.draw.rect(surface, color, rect, border_radius=4)

    # Head (drawn last, on top)
    head_rect = pygame.Rect(x_off + hx * ppb + 1, y_off + hy * ppb + 1, ppb - 2, ppb - 2)
    pygame.draw.rect(surface, COL_HEAD, head_rect, border_radius=4)

    # Fruit
    fx, fy = state.fruit
    fruit_rect = pygame.Rect(x_off + fx * ppb + 2, y_off + fy * ppb + 2, ppb - 4, ppb - 4)
    pygame.draw.rect(surface, COL_FRUIT, fruit_rect, border_radius=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Visualize trained Snake agents.")
    p.add_argument("--agent", default="ego_merged_obst_k29",
                   help="Agent key or 'all'. Options: " + ", ".join(AGENTS.keys()))
    p.add_argument("--speed", type=int, default=15, help="Steps per second (default: 15)")
    p.add_argument("--episodes", type=int, default=3, help="Episodes per agent (default: 3)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.agent == "all":
        agent_keys = [k for k in AGENTS if k != "heuristic"]
    elif args.agent in AGENTS:
        agent_keys = [args.agent]
    else:
        print(f"Unknown agent: {args.agent}")
        print(f"Options: {', '.join(AGENTS.keys())}")
        sys.exit(1)

    pygame.init()

    font_big = pygame.font.SysFont("consolas", 22, bold=True)
    font_small = pygame.font.SysFont("consolas", 16)

    board_px = BOARD_SIZE * PPB

    for agent_key in agent_keys:
        info = AGENTS[agent_key]
        print(f"\n=== {info['label']} ===")

        # Load model
        model = None
        encoder = None
        if info["ckpt"] and os.path.isfile(info["ckpt"]):
            ckpt = torch.load(info["ckpt"], map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            encoder = make_encoder(cfg["representation"], cfg["board_size"],
                                   cfg["max_length"], cfg.get("local_window_size"))
            model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            print(f"  Loaded from {info['ckpt']} (dim={encoder.output_dim})")
        else:
            print(f"  No checkpoint, using heuristic")

        # Layout: single agent fullscreen
        win_w = board_px + 2
        win_h = board_px + HEADER_H + 2
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(f"Snake Agent: {info['label']}")

        clock = pygame.time.Clock()
        paused = False
        speed = args.speed
        step_accum = 0.0

        for ep in range(args.episodes):
            env = SnakeEnv(board_size=BOARD_SIZE, max_length=100,
                           max_steps=2000, headless=True, seed=args.seed + ep)
            state = env.reset()
            done = False
            step_count = 0
            score = 0

            while not done:
                dt = clock.tick(FPS) / 1000.0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        if event.key == pygame.K_UP:
                            speed = min(speed + 5, 120)
                        if event.key == pygame.K_DOWN:
                            speed = max(speed - 5, 1)
                        if event.key == pygame.K_n:
                            done = True

                if paused:
                    # Draw paused state
                    screen.fill(COL_BG)
                    draw_board(screen, state, 1, HEADER_H, True, BOARD_SIZE, PPB)
                    txt = font_big.render("PAUSED", True, COL_SCORE)
                    screen.blit(txt, (win_w // 2 - txt.get_width() // 2, 10))
                    pygame.display.flip()
                    continue

                # Step
                step_accum += dt * speed
                while step_accum >= 1.0 and not done:
                    step_accum -= 1.0

                    if model is not None:
                        with torch.no_grad():
                            sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
                            action = torch.argmax(model(sv)).item()
                    else:
                        action = heuristic_action(env, state)

                    state, _, term, trunc, info_dict = env.step(action)
                    step_count += 1
                    score = info_dict.get("score", 0)
                    done = term or trunc

                # Draw
                screen.fill(COL_BG)
                alive = not done
                draw_board(screen, state, 1, HEADER_H, alive, BOARD_SIZE, PPB)

                # Header
                death = info_dict.get("death_reason", "") if done else ""
                status = f"DEAD ({death})" if done else "ALIVE"
                col = COL_SCORE if not done else COL_DEAD
                header = font_big.render(
                    f"{info['label']}  Ep {ep+1}/{args.episodes}  "
                    f"Score:{score}  Steps:{step_count}  [{status}]",
                    True, col
                )
                screen.blit(header, (8, 10))

                hint = font_small.render(
                    f"Speed:{speed}  SPACE=pause  UP/DOWN=speed  N=next  ESC=quit",
                    True, (120, 120, 120)
                )
                screen.blit(hint, (8, 32))

                pygame.display.flip()

            print(f"  Episode {ep+1}: score={score}, steps={step_count}, death={death}")

    pygame.quit()


if __name__ == "__main__":
    main()
