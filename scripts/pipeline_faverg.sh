#!/usr/bin/env bash
# Full training pipeline for a big CPU server:
#   1. solver-labelled dataset (20k puzzles, all families)
#   2. supervised pretraining: h64/6 layers and h128/8 layers in parallel
#   3. DAgger rounds (solver labels on the policy's own mistakes) from the better one
#   4. PPO fine-tuning with the all-family curriculum + validation / best.pt
#   5. frozen benchmark (val, then test) over every model incl. the old PPO baseline
# Usage: nohup bash scripts/pipeline_faverg.sh > runs/pipe/nohup.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
P=${PYTHON:-$HOME/venvs/faniszip/bin/python}
W=${WORKERS:-48}          # worker processes for generation / labelling
T=${THREADS:-16}          # torch threads per training process
N=${N_PUZZLES:-20000}
PPO_MIN=${PPO_MINUTES:-300}
mkdir -p runs/pipe data/imitation checkpoints
log() { echo "[$(date '+%F %T')] $*" | tee -a runs/pipe/pipeline.log; }

if [ ! -f data/imitation/ds20k.pkl ]; then
  log "1/5 generate $N puzzles (label-all, $W workers)"
  $P -m zipsolve.rl.imitation generate --n "$N" --workers "$W" --label-all --label-time 0.3 \
     --out data/imitation/ds20k.pkl > runs/pipe/gen.log 2>&1
fi

log "2/5 pretrain h64 and h128 in parallel"
[ -f checkpoints/imit20k.pt ] || $P -m zipsolve.rl.imitation train --data data/imitation/ds20k.pkl \
   --epochs 20 --batch 256 --lr 1e-3 --workers 16 --threads "$T" \
   --out checkpoints/imit20k.pt > runs/pipe/train_h64.log 2>&1 &
[ -f checkpoints/imit20k_h128.pt ] || $P -m zipsolve.rl.imitation train --data data/imitation/ds20k.pkl \
   --epochs 20 --batch 256 --lr 1e-3 --workers 16 --threads "$T" --hidden 128 --layers 8 \
   --out checkpoints/imit20k_h128.pt > runs/pipe/train_h128.log 2>&1 &
wait

BEST=$($P - <<'EOF'
from zipsolve.rl.gnn import load_model
def score(p):
    _, ck = load_model(p)
    v = ck.get("val") or {}
    return v.get("val_greedy") or 0.0
c = {p: score(p) for p in ["checkpoints/imit20k.pt", "checkpoints/imit20k_h128.pt"]}
print(max(c, key=c.get))
EOF
)
log "   pretrained best by val greedy: $BEST"

log "3/5 DAgger 4 rounds from $BEST"
[ -f checkpoints/imit20k_dagger_best.pt ] || $P -m zipsolve.rl.imitation dagger --ckpt "$BEST" \
   --data data/imitation/ds20k.pkl --rounds 4 --puzzles-per-round 5000 --samples 2 --label-time 0.3 \
   --epochs 3 --workers "$W" --threads "$T" --out checkpoints/imit20k_dagger > runs/pipe/dagger.log 2>&1

log "4/5 PPO fine-tune from DAgger best ($PPO_MIN min)"
$P -m zipsolve.rl.train --init checkpoints/imit20k_dagger_best.pt --run-name D_ppo_ft \
   --minutes "$PPO_MIN" --threads "$T" --num-envs 64 --gen-workers 4 --val-every 10 \
   > runs/D_ppo_ft.out 2>&1

log "5/5 benchmark"
CK=(--ckpt ppo_old=checkpoints/curric_4to6_final.pt --ckpt imit_h64=checkpoints/imit20k.pt
    --ckpt imit_h128=checkpoints/imit20k_h128.pt --ckpt dagger=checkpoints/imit20k_dagger_best.pt
    --ckpt ppo_ft=checkpoints/D_ppo_ft_best.pt)
for r in A_2d_long B_mixed_3d C_big_scratch; do
  [ -f "checkpoints/${r}_final.pt" ] && CK+=(--ckpt "$r=checkpoints/${r}_final.pt")
done
$P -m zipsolve.rl.benchmark run "${CK[@]}" --set val --threads 32 --out runs/bench > runs/pipe/bench_val.log 2>&1
$P -m zipsolve.rl.benchmark run "${CK[@]}" --set test --threads 32 --out runs/bench > runs/pipe/bench_test.log 2>&1
log "done"
