// sokoban-solver: a push-optimal Sokoban solver for make.fun's sokoban-solver-01.
//
// How it works. A drawn level is a hallway with independent rooms off it (a box
// pushed out of a door into the hallway can never come back, so each room is its
// own problem and the level's optimum is the sum of the rooms'). So the solver
// splits the level into rooms, solves each room push-optimally, and joins the
// move strings; walking between rooms costs no pushes.
//
// Each room is solved by A* over pushes. A state is the boxes' squares plus the
// player's reachable region (named by its lowest square): the player walks for
// free inside a region, so only pushes change the state. The heuristic is the
// minimum-cost matching of boxes to goals, where the cost of one box to one goal
// is their shortest-path distance over open floor; that is a lower bound on the
// pushes still needed, so the first solved state A* pops is push-optimal.
// Deadlocks are pruned, never by guessing: a box on a dead square (no push,
// ignoring the other boxes, can ever bring it to a goal) and a frozen 2x2 block
// of walls and boxes with a box off its goal. A* is over work, never the clock:
// every budget below counts states, so two runs write the same bytes.
//
// If the optimal search ever gives up on a room, the solver falls back to a
// weighted A* (weight 3: still complete, usually fast, not necessarily optimal)
// and then to a plain move-A* on that room, so a level is always answered with
// a valid solution when one is found at all.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four steps: [dx, dy, step letter, push letter].
const DIRS: Array<[number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budgets, in states: never the clock, so the run is deterministic.
const ASTAR_STATES = 300_000;
const GREEDY_STATES = 300_000;
const MOVE_ASTAR_EXPANSIONS = 200_000;
// The whole run's work budget: past it, searches give up at once, so a
// pathological room can never eat the 60-second wall and void the whole set.
let work = 0;
const WORK_LIMIT = 3_000_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(w: number, h: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(w * h);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    const xx = x % w;
    const xy = (x / w) | 0;
    for (const [dx, dy] of DIRS) {
      const yx = xx - dx;
      const yy = xy - dy;
      if (yx < 0 || yy < 0 || yx >= w || yy >= h) continue;
      const y = yy * w + yx;
      const bx = xx - 2 * dx;
      const by = xy - 2 * dy;
      if (bx < 0 || by < 0 || bx >= w || by >= h) continue;
      const behind = by * w + bx;
      if (!wall[y] && !wall[behind] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * The squares the player reaches from `from` with `occupied` in the way, marked in `seen` with `stamp`; returns the
 * region's lowest square. `stack` is scratch space of the board's size.
 */
function reach(w: number, h: number, wall: Uint8Array, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const px = p % w;
    const py = (p / w) | 0;
    for (const [dx, dy] of DIRS) {
      const qx = px + dx;
      const qy = py + dy;
      if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
      const q = qy * w + qx;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(w: number, h: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = w * h;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    const px = p % w;
    const py = (p / w) | 0;
    for (const [dx, dy, letter] of DIRS) {
      const qx = px + dx;
      const qy = py + dy;
      if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
      const q = qy * w + qx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** All permutations of 0..n-1, for the box-to-goal matching (rooms hold at most 4 boxes). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const gen = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      gen();
      cur.pop();
      used[i] = false;
    }
  };
  gen();
  return out;
}

// A state's key: the region's lowest square and the boxes' squares (exact while size ** (boxes + 1) stays under 2 ** 53).
function stateKey(boxes: number[], low: number, size: number): number | string {
  if (boxes.length <= 5) {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  }
  return `${low}:${boxes.join(',')}`;
}

/**
 * The push-optimal (weight 1) or greedy (weight > 1) solution of `sub`, searching at most `maxStates` states:
 * {pushes, moves}, or null when the search gives up. A state is the boxes and the player's region; the search is A*
 * on pushes with the min-cost box-to-goal matching as its estimate, pruning dead squares and frozen 2x2 blocks.
 */
function searchPushes(sub: Level, weight: number, maxStates: number): { pushes: number; moves: string } | null {
  const { w, h, wall, goals, boxes, player } = sub;
  const size = w * h;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, h, wall, goals);
  if (boxes.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (solved(boxes)) return { pushes: 0, moves: '' };
  // The distance from every square to every goal, over open floor (boxes ignored): the matching's costs.
  const distG: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    d[g] = 0;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi];
      const px = p % w;
      const py = (p / w) | 0;
      for (const [dx, dy] of DIRS) {
        const qx = px + dx;
        const qy = py + dy;
        if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
        const n = qy * w + qx;
        if (wall[n] || d[n] !== -1) continue;
        d[n] = d[p] + 1;
        q.push(n);
      }
    }
    return d;
  });
  const perms = boxes.length <= 4 ? permutations(boxes.length) : null;
  // The estimate depends only on the boxes, and one box set recurs with different player regions: memoize it.
  const hCache = new Map<string, number>();
  // The estimate: the cheapest way to put every box on a distinct goal (exact matching for small rooms, the sum of
  // each box's nearest goal otherwise). Both are lower bounds on the pushes still needed.
  const estimate = (bs: number[]): number => {
    const ck = bs.join(',');
    const hit = hCache.get(ck);
    if (hit !== undefined) return hit;
    let out: number;
    if (perms !== null) {
      let best = Infinity;
      for (const perm of perms) {
        let s = 0;
        for (let i = 0; i < bs.length; i++) {
          const d = distG[perm[i]]![bs[i]]!;
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
      out = best;
    } else {
      let s = 0;
      for (const b of bs) {
        let m = Infinity;
        for (const d of distG) {
          const v = d[b]!;
          if (v >= 0 && v < m) m = v;
        }
        if (m === Infinity) {
          out = Infinity;
          hCache.set(ck, out);
          return out;
        }
        s += m;
      }
      out = s;
    }
    hCache.set(ck, out);
    return out;
  };
  if (estimate(boxes) === Infinity) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;
  const statesBoxes: number[][] = [boxes.slice()];
  const statesPlayer: number[] = [player];
  const statesG: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const closed = new Map<number | string, number>();
  for (const b of boxes) occupied[b] = 1;
  const startLow = reach(w, h, wall, occupied, player, seen, ++stamp, stack);
  for (const b of boxes) occupied[b] = 0;
  closed.set(stateKey(boxes, startLow, size), 0);
  // The open list, a heap on f = g + weight * h, ties to the deeper state first, then to the earlier one: fixed order.
  const heap: Array<[number, number, number, number]> = [[estimate(boxes), 0, 0, 0]];
  let seq = 1;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let found = -1;
  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    const bs = statesBoxes[s]!;
    for (const b of bs) occupied[b] = 1;
    const low = reach(w, h, wall, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    if (closed.get(stateKey(bs, low, size)) !== g) {
      for (const x of bs) occupied[x] = 0;
      continue;
    }
    if (solved(bs)) {
      for (const x of bs) occupied[x] = 0;
      found = s;
      break;
    }
    const here = stamp;
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (const [dx, dy] of DIRS) {
        const px = bx - dx;
        const py = by - dy;
        if (px < 0 || py < 0 || px >= w || py >= h) continue;
        const pushSq = py * w + px;
        const tx = bx + dx;
        const ty = by + dy;
        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
        const t = ty * w + tx;
        if (seen[pushSq] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key: number | string = '';
        let next: number[] = [];
        if (!frozen) {
          next = bs.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const nextLow = reach(w, h, wall, occupied, b, mark, ++markStamp, stack);
          key = stateKey(next, nextLow, size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const prev = closed.get(key);
        if (prev !== undefined && prev <= g + 1) continue;
        if (statesBoxes.length >= maxStates || work >= WORK_LIMIT) {
          for (const x of bs) occupied[x] = 0;
          return null;
        }
        work++;
        closed.set(key, g + 1);
        const idx = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        statesG.push(g + 1);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        const hEst = estimate(next);
        if (hEst === Infinity) continue;
        heapPush([g + 1 + weight * hEst, g + 1, seq++, idx]);
      }
    }
    for (const x of bs) occupied[x] = 0;
  }
  if (found < 0) return null;
  // The path of pushes, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let curPlayer = player;
  let curBoxes = boxes.slice();
  try {
    for (const s of path) {
      const from = pushFrom[s]!;
      const to = pushTo[s]!;
      const d = to - from;
      const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
      for (const b of curBoxes) occupied[b] = 1;
      moves += walkTo(w, h, wall, occupied, curPlayer, from - d);
      for (const b of curBoxes) occupied[b] = 0;
      moves += dir[3];
      curPlayer = from;
      curBoxes = curBoxes.slice();
      curBoxes[curBoxes.indexOf(from)] = to;
      curBoxes.sort((a, b) => a - b);
    }
  } catch {
    return null;
  }
  return { pushes: path.length, moves };
}

/**
 * A plain move-A* on one room (the house baseline's shape, with true floor distances): the last resort when the push
 * searches give up. Fewest moves, not fewest pushes, but a valid solution beats none.
 */
function searchMoves(sub: Level, maxExpansions: number): { pushes: number; moves: string } | null {
  const { w, h, wall, goals, boxes, player } = sub;
  const size = w * h;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dist = (b: number): number => {
    let m = Infinity;
    for (const g of goals) {
      const d = Math.abs((g % w) - (b % w)) + Math.abs(((g / w) | 0) - ((b / w) | 0));
      if (d < m) m = d;
    }
    return m;
  };
  const est = (bs: number[]): number => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p: number, bs: number[]): string => `${p}:${bs.join(',')}`;
  const sp: number[] = [player];
  const sb: number[][] = [boxes.slice()];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(player, boxes), 0]]);
  const heap: Array<[number, number, number]> = [[est(boxes), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < maxExpansions && work < WORK_LIMIT) {
    const [, g, s] = heapPop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goal[b])) {
      let moves = '';
      let pushes = 0;
      for (let t = s; t > 0; t = parent[t]!) {
        moves = step[t] + moves;
        const c = step[t]!;
        if (c >= 'A' && c <= 'Z') pushes++;
      }
      return { pushes, moves };
    }
    expanded++;
    work++;
    const p = sp[s]!;
    const px = p % w;
    const py = (p / w) | 0;
    for (const [dx, dy, letter, upper] of DIRS) {
      const qx = px + dx;
      const qy = py + dy;
      if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
      const q = qy * w + qx;
      if (wall[q]) continue;
      let nb = bs;
      let move = letter;
      const at = bs.indexOf(q);
      if (at >= 0) {
        const tx = qx + dx;
        const ty = qy + dy;
        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
        const t = ty * w + tx;
        if (wall[t] || bs.includes(t)) continue;
        nb = bs.slice();
        nb[at] = t;
        nb.sort((a, b) => a - b);
        move = upper;
      }
      const k = key(q, nb);
      const seenIdx = best.get(k);
      if (seenIdx !== undefined && sg[seenIdx]! <= g + 1) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      parent.push(s);
      step.push(move);
      best.set(k, t);
      heapPush([g + 1 + est(nb), g + 1, t]);
    }
  }
  return null;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal) out of the
 * floor, and each part left that holds a box is a room, its squares, boxes and goals. Null when some part holds more
 * or fewer boxes than goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Array<{ squares: number[]; boxes: number[]; goals: number[]; done?: boolean }> | null {
  const { w, h, wall, goals, boxes, player } = level;
  const live = liveSquares(w, h, wall, goals);
  const row = (player / w) | 0;
  const cut = new Uint8Array(w * h);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(w * h).fill(-1);
  const rooms: Array<{ squares: number[]; boxes: number[]; goals: number[]; done?: boolean }> = [];
  for (let s = 0; s < w * h; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      const px = p % w;
      const py = (p / w) | 0;
      for (const [dx, dy] of DIRS) {
        const qx = px + dx;
        const qy = py + dy;
        if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
        const q = qy * w + qx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The level solved room by room: each room's push-optimal solution in turn, the player walking on from where the last room left it. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const sol = searchPushes(level, 1, ASTAR_STATES) ?? searchPushes(level, 2, GREEDY_STATES) ?? searchMoves(level, MOVE_ASTAR_EXPANSIONS);
    return sol === null ? '' : sol.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const sol = searchPushes(sub, 1, ASTAR_STATES) ?? searchPushes(sub, 3, GREEDY_STATES) ?? searchMoves(sub, MOVE_ASTAR_EXPANSIONS);
    if (sol === null) return '';
    moves += sol.moves;
    player = endOf(level.w, player, sol.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
