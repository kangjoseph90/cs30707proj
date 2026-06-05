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
    "target_gamma099": {
        "rep": "egocentric_merged_obstacle_local",
        "ws": 29,
        "ckpt": "results/sweep_500k_target_dqn_gamma099/egocentric_merged_obstacle_local_w29_seed0/checkpoint.pt",
        "label": "Target γ=0.99",
    },
    "target_gamma095": {
        "rep": "egocentric_merged_obstacle_local",
        "ws": 29,
        "ckpt": "results/sweep_500k_target_dqn/egocentric_merged_obstacle_local_w29_seed0/checkpoint.pt",
        "label": "Target γ=0.95",
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
    
    # Planner args
    p.add_argument("--use-planner", action="store_true", default=False)
    p.add_argument("--planner-weight", type=float, default=2.0)
    p.add_argument("--planner-area-weight", type=float, default=1.0)
    p.add_argument("--planner-tail-reach-weight", type=float, default=1.0)
    
    # Custom checkpoint loading
    p.add_argument("--checkpoint", type=str, default=None, help="Path to custom checkpoint.pt file to load")
    
    args = p.parse_args()

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        AGENTS["custom"] = {
            "rep": cfg["representation"],
            "ws": cfg.get("local_window_size"),
            "ckpt": args.checkpoint,
            "label": "Custom Checkpoint",
        }
        agent_keys = ["custom"]
    elif args.agent == "all":
        agent_keys = [k for k in AGENTS if k != "heuristic" and k != "custom"]
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

    if args.use_planner:
        from planner import SafetyPlanner
        planner = SafetyPlanner(
            planner_weight=args.planner_weight,
            area_weight=args.planner_area_weight,
            tail_reach_weight=args.planner_tail_reach_weight,
        )
    else:
        planner = None

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

        # Layout: single agent fullscreen + planner info space if using planner
        win_w = board_px + 2
        extra_h = 160 if args.use_planner else 0
        win_h = board_px + HEADER_H + extra_h + 2
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
            
            last_decision = None
            last_q_values = None

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
                            q_values = model(sv).squeeze(0).numpy()
                        if args.use_planner:
                            last_decision = planner.choose_action(env, q_values)
                            action = last_decision.chosen_action
                            last_q_values = q_values
                        else:
                            action = np.argmax(q_values).item()
                            last_decision = None
                            last_q_values = q_values
                    else:
                        action = heuristic_action(env, state)
                        last_decision = None
                        last_q_values = None

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

                # Safety planner panel drawing
                if args.use_planner and last_decision is not None:
                    y_start = board_px + HEADER_H
                    pygame.draw.line(screen, (80, 80, 90), (0, y_start), (win_w, y_start), 2)
                    
                    panel_rect = pygame.Rect(1, y_start + 2, win_w - 2, 158)
                    pygame.draw.rect(screen, (20, 20, 25), panel_rect)
                    
                    title = font_small.render(
                        f"SAFETY PLANNER (W_p={args.planner_weight:.1f})",
                        True, (0, 180, 255)
                    )
                    screen.blit(title, (12, y_start + 8))
                    
                    dqn_argmax = np.argmax(last_q_values) if last_q_values is not None else -1
                    if last_decision.chosen_action != dqn_argmax:
                        override_txt = font_small.render("PLANNER OVERRIDE ACTIVE", True, (255, 140, 0))
                        screen.blit(override_txt, (250, y_start + 8))
                        
                    actions_labels = ["LEFT (0)", "STRAIGHT (1)", "RIGHT (2)"]
                    col_width = 148
                    start_x = 12
                    
                    for a in range(3):
                        info_act = last_decision.actions[a]
                        x_pos = start_x + a * 154
                        
                        col_rect = pygame.Rect(x_pos, y_start + 30, col_width, 118)
                        if a == last_decision.chosen_action:
                            is_override = (last_decision.chosen_action != dqn_argmax)
                            border_color = (255, 140, 0) if is_override else (0, 255, 120)
                            pygame.draw.rect(screen, (30, 45, 35) if not is_override else (45, 35, 25), col_rect)
                            pygame.draw.rect(screen, border_color, col_rect, 2)
                            
                            chosen_txt = font_small.render("[SELECTED]", True, border_color)
                            screen.blit(chosen_txt, (x_pos + 8, y_start + 34))
                        else:
                            pygame.draw.rect(screen, (25, 25, 30), col_rect)
                            pygame.draw.rect(screen, (60, 60, 70), col_rect, 1)
                            
                            act_txt = font_small.render(actions_labels[a], True, (150, 150, 150))
                            screen.blit(act_txt, (x_pos + 8, y_start + 34))
                            
                        if info_act.immediate_death:
                            death_txt = font_small.render("DEATH / MASKED", True, COL_DEAD)
                            screen.blit(death_txt, (x_pos + 8, y_start + 55))
                            score_txt = font_small.render("Score: -inf", True, (120, 120, 120))
                            screen.blit(score_txt, (x_pos + 8, y_start + 115))
                        else:
                            q_txt = font_small.render(f"Q-val: {info_act.q_value:.2f}", True, (200, 200, 200))
                            screen.blit(q_txt, (x_pos + 8, y_start + 55))
                            
                            area_pct = info_act.reachable_area / (BOARD_SIZE * BOARD_SIZE) * 100
                            area_txt = font_small.render(f"Area: {area_pct:.1f}% ({info_act.reachable_area})", True, (180, 180, 200))
                            screen.blit(area_txt, (x_pos + 8, y_start + 75))
                            
                            tail_str = "Tail: REACHABLE" if info_act.can_reach_tail else "Tail: BLOCKED"
                            tail_col = (100, 255, 100) if info_act.can_reach_tail else (200, 100, 100)
                            tail_txt = font_small.render(tail_str, True, tail_col)
                            screen.blit(tail_txt, (x_pos + 8, y_start + 95))
                            
                            score_txt = font_small.render(f"Score: {info_act.combined_score:.2f}", True, COL_SCORE)
                            screen.blit(score_txt, (x_pos + 8, y_start + 115))

                pygame.display.flip()

            print(f"  Episode {ep+1}: score={score}, steps={step_count}, death={death}")

    pygame.quit()


if __name__ == "__main__":
    main()
