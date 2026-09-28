# Puzzle bank (format v1)

Curated, pre-rated puzzles shared by the local app (`python -m zipsolve.app`) and
the static site. Built by `scripts/build_bank.py` (resumable; see its docstring).
Every puzzle has exactly one solution (generator `unique=True`, re-verified with
`count_solutions(p, 2) == (1, "complete")`), stored in `puzzle.solution`, so
hints / "Show me" can use it directly. JSON files are compact (no whitespace)
except `index.json`.

## index.json

```json
{"version": 1, "generated": "<iso utc>", "daily_start": "2026-01-01", "daily_end": "2027-12-31",
 "daily_epoch": "2026-01-01", "labels": ["Easy","Medium","Hard","Expert","Insane"],
 "robots": "robots/{id}.json", "daily_files": "daily/{yyyy}-{mm}.json", "puzzles": 3300,
 "modes": {"classic": {"easy": {"file": "pools/classic-easy.json", "count": 80}, "medium": {...}, "hard": {...}},
           "walls": {...}, "islands": {...}, "cube": {...}}}
```

Modes: `classic` (plain grid), `walls`, `islands` (organic archipelago with
bridges), `cube` (3D). Difficulties: `easy`, `medium`, `hard`.

## Entries (pools and dailies)

`pools/<mode>-<diff>.json` is a list of entries, sorted by id:

```json
{"id": "c1a2b3c4d5", "mode": "classic", "diff": "hard", "label": "Expert", "score": 67.3,
 "size": "8x8", "puzzle": {"kind", "coords", "edges", "meta", "checkpoints", "solution"}}
```

* `id`: mode letter (`c` classic, `w` walls, `i` islands, `q` cube) + 9 hex digits
  of a hash of the puzzle. Unique across the whole bank (pools and dailies never
  share a puzzle).
* `label` / `score`: the human difficulty from `zipsolve.difficulty.rate`
  (0-100; Easy < 22 <= Medium < 42 <= Hard < 62 <= Expert < 80 <= Insane).
  `diff` is the bucket; the label is the truthful rating (a `hard` pool holds
  Hard and Expert puzzles; cube buckets are relative: easy ~ Medium, hard ~ Insane).
* `size`: display text (`"7x7"`, `"4 islands"`, `"3x3x4"`).
* `puzzle`: `Puzzle.to_dict()` (coords as integers), including `solution`.

**Picking from a pool** (so local and static agree on `#play=...&seed=` links):
`entry = pool[seed % pool.length]`. The game's mode buttons map to pools via
`bank` in `play/modes.js` (`/api/generate` body field `bank: "<mode>-<diff>"`):
classic easy/medium/hard -> `classic-*`, Walls -> `walls-medium`,
Islands -> `islands-medium`, 3D Cube -> `cube-hard`, the race -> `classic-medium`.

## Daily schedule

`daily/<yyyy>-<mm>.json`:

```json
{"2026-01-01": {"number": 1, "easy": <entry>, "medium": <entry>, "hard": <entry>, "special": null},
 "2026-01-04": {"number": 4, ..., "special": <entry>}}
```

* One record per date from `daily_start` through `daily_end`;
  `number` = days since 2026-01-01 + 1 (the game's "Zip #n").
* `easy` / `medium` / `hard` are classic puzzles rated Easy / Medium /
  Hard-or-Expert, not boring, with well spread numbers; within each ISO week
  they ramp up (Monday easiest, Sunday hardest).
* `special` is set on Sundays only (else `null`): rotating islands -> walls ->
  cube (islands/walls from the `hard` criteria, cube from `medium`).
* The local server's `/api/daily?difficulty=easy|medium|hard|special&date=`
  serves exactly these (same `daily` block: `{date, number, difficulty, label}`),
  plus `rating: {label, score, mode, diff}`.

## Robot runs

`robots/<id>.json` (one per bank puzzle, fetched lazily):

```json
{"model": "zip_gnn.pt",
 "rookie":      {...},  // /api/solve/rl mode "greedy" (no trace)
 "scout":       {...},  // /api/solve/rl mode "search", trace
 "grandmaster": {...},  // /api/solve/rl mode "hybrid", trace
 "tortoise":    {...}}  // /api/solve/exact with trace (plain exact solver)
```

Each run has the app's response shape (minus the per-step top-3 lists):
`mode`, `status` (`solved` / `stuck` / `unsat` / `budget` / `timeout`), `solved`,
`path` (the solution, or the deepest / stuck partial path), `start_len`,
`nodes_expanded`, `backtracks`, `seconds` (Python run time), `stuck_at` and
`dead_from` when not solved, `mean_confidence`, `min_confidence`, and
`steps: [{"p": 0.93}, ...]` (the policy's probability of each move along
`path`; rookie/scout/grandmaster only). Runs were made with the app's settings:
budget 4000, 20 s time limit.

`trace` (scout, grandmaster, tortoise) uses the app's compact event encoding:
a node id (>= 0) = push that node, a negative number `-k` = pop `k` nodes,
`"R"` = a new search attempt starts (clear back to checkpoint 1). Traces are
capped at 4000 raw events; then `trace_truncated: true` and the replay should
finish on `path` (as show.js already does).

### Strategy robots (zipsolve.robots)

Some robot files also hold runs of the strategy robots, keyed by robot id:
`detective`, `mcts` (Sage), `evolver`, `gambler`, `sat` (Mathematician) - for
every "more modes" puzzle and for the pools the AI show picks from most
(`classic-medium`, `classic-hard`, `walls-medium`, `islands-medium`,
`architect-expert`; elsewhere the online show says the robot "hasn't studied" it).
Built by `scripts/build_bank_modes.py` (8 s time limit, seed = id digits % 1000),
in the `/api/solve/robot` shape but compacted: `robot_name` / `emoji` are left
out, `steps` is stored as parallel lists `sp` (p per move) and `sh` (Detective
technique per move) along `path[1:]`, Evolver generations and Sage tree
snapshots are thinned to ~12 each (Sage trees: top 3 children with 2 kids each),
notes are capped per kind (the Detective's notes explaining the solution are kept),
`stats.fitness_history` to 40 points. Traces end on `path` (a final
`{"t":"path"}` event is added when needed). `play/static_api.js` expands them
back. The Detective's notes along the unique solution double as the online
"teaching hints" (`/api/hint/explain`).

## More modes (scripts/build_bank_modes.py, additive)

`pools/<mode>-medium.json` and `pools/<mode>-hard.json` for `portals`, `torus`,
`hex`, `tri`, `oneway`, `overpass`, `keys`, `cubesurf` (50 + 40 puzzles), and
`pools/coop-medium.json` (40 two-path co-op puzzles, 6x6). Ids: `p`, `t`, `h`,
`r`, `o`, `v`, `k`, `s`, `x` + 9 hex digits. Entry `puzzle` is the kind's
`to_dict()`; one-way arcs / key-door precedence are top-level `arcs` /
`precedence`. Co-op entries: `CoopPuzzle.to_dict(include_solution=True)`, so
the two unique paths are under `meta.coop.solution.paths` (served without it).
`index.json` gets `modes.<mode>` entries plus `more_modes: {modes, puzzles,
robots, generated}`. `scripts/build_bank.py` keeps other builders' modes and
index keys when it rewrites index.json.

## The Architect

`pools/architect-expert.json`, `pools/architect-insane.json` and
`architect/weekly.json` (`{"YYYY-Www": entry}`, ISO weeks 2026-W01 .. 2027-W52)
are built by `scripts/architect_designs.py`. The game's home shows the week's
puzzle ("This week's Architect challenge": `GET /api/architect/weekly?week=`;
weeks outside the file wrap around: key = sorted keys[(year * 53 + week) % n]).
Online, "Ask the Architect" picks one of its certified designs from the pools.
