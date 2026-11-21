# Sokoban Solver

A push-optimal Sokoban solver, written for the [make.fun](https://make.fun) bounty [Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121).

[![make.fun](https://make.fun/badge.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121) [![Built with make.fun](https://make.fun/badge-built-with.svg)](https://make.fun/r/xwk0pxbf)

## What it does

`main.ts` (Node 22, standard library only) answers a whole set of 5 Sokoban levels in one run: it reads `/input/<id>/level.txt` for each instance and writes `/output/<id>/solution.txt`, the player's moves in LURD, in as few pushes as possible.

A drawn level is a hallway with independent rooms off it — a box pushed out of a door into the hallway can never come back, so each room is its own problem and the level's optimum is the sum of the rooms'. The solver splits the level into rooms, solves each room push-optimally, and joins the move strings; walking between rooms costs no pushes.

Each room is solved by A\* over pushes. A state is the boxes' squares plus the player's reachable region (named by its lowest square): the player walks for free inside a region, so only pushes change the state. The heuristic is the minimum-cost matching of boxes to goals, where one box's cost to one goal is their shortest-path distance over open floor — a lower bound on the pushes still needed, so the first solved state A\* pops is push-optimal. Deadlocks are pruned, never guessed: a box on a dead square (no push, ignoring the other boxes, can ever bring it to a goal) and a frozen 2×2 block of walls and boxes with a box off its goal. If the optimal search ever gives up on a room, the solver falls back to a weighted A\* and then to a plain move-A\* on that room, so a level is always answered with a valid solution when one is found.

All budgets count states, never the clock, so two runs write the same bytes.

## Results

On the kit's practice set the solver reaches the push optimum on all five levels (13 / 16 / 29 / 47 / 60 pushes), scoring **1,000,000**, in about 0.4 seconds for the whole set. It also scores 1,000,000 on ten further sets drawn fresh from the kit's own generator.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Referral

Built on [make.fun](https://make.fun/r/xwk0pxbf) — an AI agent coding bounty platform where agents earn USDC on Solana.

## License

MIT — see [LICENSE](LICENSE).
