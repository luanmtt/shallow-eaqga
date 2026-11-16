# Scripts de simulação noturna

Copiados de `~/projects/qga/quantum/scripts`.

- `night_sim.sh`: monitor genérico. Inicia (ou acompanha) um comando, espera o
  fim e pode desligar o PC. Serve para o `eaqga.py` via `-- <comando>`.
- `run_night.sh` / `run_retry.sh`: específicos do repositório QGA antigo
  (`qga.py`, benchmark functions, `.qga`). Ficam como referência; **não** são
  adaptados ao `eaqga.py`.

Exemplo para o E1-S40 noturno:

```bash
./scripts/night_sim.sh -t 6h --no-poweroff -- \
  .venv/bin/python eaqga.py --preset paper \
  --sizes 40 --population 10 20 --subsets 10 --runs 5 \
  --algorithms baseline ours original --original-runs 5 \
  --shots-original 20 --no-fixed-h --jobs 6 \
  --save-table --output outputs/plannedComparisons/E1-S40
```

Use `--poweroff` no lugar de `--no-poweroff` se quiser desligar ao terminar.
