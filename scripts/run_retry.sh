#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_retry.sh — re-roda jobs específicos (pós-crash) e funde o resultado nos
# batches canônicos.
#
# Uso (com o monitor night_sim.sh — a pergunta do poweroff é feita no INÍCIO):
#   cd quantum
#   ./scripts/night_sim.sh -t 10h -- ./scripts/run_retry.sh 100 --jobs 8
#
# Uso sem monitor (sem poweroff):
#   ./scripts/run_retry.sh 100 --jobs 8
#
# Args posicionais:
#   1º  N_SIMS   (default 100)
#   demais       flags repassadas ao qga.py (ex: --jobs 8)
#
# Jobs (função:tipos):
#   sphere:quantum, ackley:quantum → classical canônico copiado p/ o dir
#     fresco (vira dataset completo em SIM_<hora-dia>/<função>/; os
#     BATCH_* antigos não são tocados)
#   schwefel:classical,quantum     → batch novo, sem merge
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # quantum/scripts
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                        # raiz do repo
cd "$SCRIPT_DIR/.."                                            # quantum/ (onde fica qga.py)

N_SIMS="${1:-100}"
shift 1 2>/dev/null || true
EXTRA="$*"

# usa o venv do projeto se existir, senão o python do PATH
PY="python"
[ -x "$ROOT/.qga/bin/python" ] && PY="$ROOT/.qga/bin/python"

# Teto de workers paralelos do joblib/loky (mitigação contra o SEGV no
# feeder/pickle). Override: LOKY_MAX_CPU_COUNT=16 ./scripts/run_retry.sh ...
export PYTHONFAULTHANDLER=1  # stack Python no stderr se um worker morrer
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"

# Sessão da campanha (ver run_night.sh): os batches vão p/ este SIM_<hora-dia>/.
export QGA_SIM_ID="SIM_$(date +%Hh%M-%d_%m)"

JOBS=( "sphere:quantum" "ackley:quantum" "schwefel:classical,quantum" )
# Ponte da era BATCH: classical canônico p/ jobs quantum-only, copiado p/ o
# dir fresco (que vira o dataset completo; os BATCH_* antigos não são tocados).
declare -A LEGACY_CLASSICAL=( [sphere]="outputs/BATCH_01h08/classical" [ackley]="outputs/BATCH_02h14/classical" )

for job in "${JOBS[@]}"; do
    f="${job%%:*}"; types="${job##*:}"
    echo "[run_retry] === job: $f [$types] ==="
    joblog="/tmp/retry_${f}.log"
    # shellcheck disable=SC2086
    $PY qga.py --batch "$N_SIMS" --types "$types" --function "$f" $EXTRA 2>&1 | tee "$joblog"
    code=${PIPESTATUS[0]}
    if [ "$code" -ne 0 ]; then
        echo "[run_retry] !! $f falhou (exit $code) — continuando"
        continue
    fi
    newdir=$(grep -a -o 'outputs/SIM_[0-9A-Za-z_.\-]*' "$joblog" | tail -1)
    if [ -z "$newdir" ]; then
        newdir=$(grep -a -o 'outputs/BATCH_[0-9]*h[0-9]*' "$joblog" | tail -1)
    fi
    if [ -z "$newdir" ]; then
        echo "[run_retry] !! não achei o dir do batch no log — merge pulado"
        continue
    fi
    echo "[run_retry] $f OK → $newdir"
    dest="$newdir"
    merged=0
    src="${LEGACY_CLASSICAL[$f]:-}"
    if [ -n "$src" ]; then
        # job quantum-only: trazer o classical canônico p/ o dir fresco,
        # que vira o dataset completo (só se ambos têm N_SIMS linhas)
        fresh_csv="$newdir/quantum/batch_results.csv"
        base_csv="$src/batch_results.csv"
        if [ ! -f "$fresh_csv" ] || [ ! -f "$base_csv" ]; then
            echo "[run_retry] !! CSVs p/ merge ausentes — merge pulado"
            continue
        fi
        fresh_rows=$(($(wc -l < "$fresh_csv") - 1))
        base_rows=$(($(wc -l < "$base_csv") - 1))
        if [ "$fresh_rows" -ne "$N_SIMS" ] || [ "$base_rows" -ne "$N_SIMS" ]; then
            echo "[run_retry] !! merge pulado (fresh=$fresh_rows base=$base_rows, esperado $N_SIMS)"
            continue
        fi
        echo "[run_retry] merge $src → $dest/classical"
        cp -r "$src" "$dest/classical"
        merged=1
    fi
    if [ "$merged" -eq 0 ]; then
        continue  # batch full: gráficos já gerados dentro do qga.py
    fi
    echo "[run_retry] regenerando gráficos de $dest"
    $PY -m visualization.analysis "$dest" --function "$f" --pop 50
    $PY - <<PYEOF
from pathlib import Path
from src.benchmark_functions import get_benchmark
from visualization.multi_run import animate_multi_run_2d, animate_multi_run_3d
bd = Path("$dest")
fn, domain, opt, opt_label = get_benchmark("$f", num_variaveis=2)
animate_multi_run_2d(bd, fn, domain, optimum=opt, optimum_label=opt_label, target_function="$f", save_path=bd / "multi_run_2d.gif")
animate_multi_run_3d(bd, fn, domain, optimum=opt, optimum_label=opt_label, target_function="$f", save_path=bd / "multi_run_3d.gif")
print("[run_retry] 2D/3D OK")
PYEOF
done

echo "[run_retry] fim."
