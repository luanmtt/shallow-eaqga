#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_night.sh — roda o batch em sequência para todas as funções do paper.
#
# Uso (com o monitor night_sim.sh, que desliga o PC quando terminar):
#   cd quantum
#   ./scripts/night_sim.sh -t 9h -- ./scripts/run_night.sh 100 classical,quantum
#
# Args posicionais:
#   1º  N_SIMS   (default 100)
#   2º  TYPES    (default "classical,quantum")
#   demais       flags repassadas ao qga.py (ex: --exact)
#
# Funções: rastrigin, sphere, ackley, griewank, rosenbrock (as do paper).
#
# Saída: outputs/SIM_<hora-dia>/<função>/ (agrupa a campanha p/ comparação).
# QGA_SIM_ID fixa a sessão; sem ele, cada qga.py gera o próprio.
#
# Exit codes na mensagem de falha (exit N): 139 = SIGSEGV (o crash do
# feeder/pickle do joblib), 143 = SIGTERM, 137 = SIGKILL/OOM-killer,
# 1 = erro Python ou batch parcial (sims com falha; ver coluna 'error').
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # quantum/scripts
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                        # raiz do repo
cd "$SCRIPT_DIR/.."                                            # quantum/ (onde fica qga.py)

N_SIMS="${1:-100}"
TYPES="${2:-classical,quantum}"
shift 2 2>/dev/null || true
EXTRA="$*"

# usa o venv do projeto se existir, senão o python do PATH
PY="python"
[ -x "$ROOT/.qga/bin/python" ] && PY="$ROOT/.qga/bin/python"

# Teto de workers paralelos do joblib/loky (mitigação contra o SEGV no
# feeder/pickle: menos pickles gigantes concorrentes em voo).
# Override: LOKY_MAX_CPU_COUNT=16 ./scripts/run_night.sh ...
export PYTHONFAULTHANDLER=1  # stack Python no stderr se um worker morrer
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"

# Sessão da campanha: todos os qga.py movem seu batch p/ este SIM_<hora-dia>/.
export QGA_SIM_ID="SIM_$(date +%Hh%M-%d_%m)"

FUNCTIONS=(rastrigin sphere ackley griewank rosenbrock)

for f in "${FUNCTIONS[@]}"; do
    echo "[run_night] === função: $f ==="
    # shellcheck disable=SC2086
    $PY qga.py --batch "$N_SIMS" --types "$TYPES" --function "$f" $EXTRA \
        || { code=$?; echo "[run_night] !! $f falhou (exit $code) — continuando"; }
done

echo "[run_night] fim."
