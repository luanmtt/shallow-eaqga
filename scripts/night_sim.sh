#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# night_sim.sh — monitora a simulação do QGA e desliga o PC quando ela terminar.
#
# Consumo de energia: praticamente zero.
#   • Fase 1 (até 95% do tempo estimado): dorme em blocos de 10 min — a pollagem
#     só checa se o processo ainda existe (kill -0), nada de CPU.
#   • Fase 2 (últimos 5% do tempo estimado): pola a cada 30s para detectar
#     o fim o quanto antes e desligar.
#
# Uso 1 — o script inicia a simulação (roda em background, sobrevive ao terminal via nohup):
#   ./scripts/night_sim.sh -t 5h -- python qga.py --batch 50
#
# Uso 2 — a simulação JÁ está rodando (monitora um PID):
#   ./scripts/night_sim.sh -t 5h -p 12345
#
# Uso 2b — sem PID: autodetecta pelo pgrep:
#   ./scripts/night_sim.sh -t 5h
#
# Opções:
#   -t, --time      Duração estimada  (5h | 300m | 45s | 18000 — default 5h)
#   -p, --pid       PID de um processo já em execução
#   -w, --window    Percentual de proximidade p/ começar a pollar (default 5 → 95% do tempo estimado)
#   -i, --interval  Intervalo de pollagem da fase 2 em segundos (default 30)
#   -g, --grace     Espera (s) após detectar o fim antes de desligar (default 90, dá tempo do CSV flushar)
#   -y, --yes       Desliga sem perguntar (atalho p/ --poweroff)
#   --poweroff      Desliga no fim sem perguntar (pula a pergunta inicial)
#   --no-poweroff   Não desliga no fim (pula a pergunta inicial)
#   -n, --dry-run   Não desliga de verdade (só imprime o que faria)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -uo pipefail

# ── defaults ────────────────────────────────────────────────────────────────────────────────────────
EST_SECS=18000        # 5h
TARGET_PID=""
PROXIMITY_PCT=5
POLL_INTERVAL=30
GRACE=90
POWEROFF=""   # "" = ainda não decidido (--poweroff=1, --no-poweroff=0, ou pergunta inicial)
DRY_RUN=0
DEBUG_PARSE=0
CMD=()

log() { printf '[night_sim %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

die() { log "ERRO: $*"; exit 1; }

# ── parse args ──────────────────────────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--time)
            t="${2:?}"
            case "$t" in
                # formato composto: 7h46, 1h30m, 2h, 300m, 45s, segundos puros
                *h*|*m*|*s*)
                    if [[ $t =~ ^([0-9]+)h([0-9]*)(m?)([0-9]*)(s?)$ ]]; then
                        _h=${BASH_REMATCH[1]:-0}
                        _m=${BASH_REMATCH[2]:-0}
                        _s=${BASH_REMATCH[4]:-0}
                        EST_SECS=$(( _h*3600 + _m*60 + _s ))
                    elif [[ $t =~ ^([0-9]+)m$ ]]; then
                        EST_SECS=$(( ${BASH_REMATCH[1]} * 60 ))
                    elif [[ $t =~ ^([0-9]+)s$ ]]; then
                        EST_SECS=${BASH_REMATCH[1]}
                    else
                        die "-t formato inválido: '$t' (use ex. 5h, 300m, 45s, 7h46 ou 18000)"
                    fi
                    ;;
                *[!0-9]*) die "-t formato inválido: '$t' (use ex. 5h, 300m, 45s, 7h46 ou 18000)" ;;
                *)  EST_SECS=$t ;;
            esac
            shift 2 ;;
        -p|--pid)    TARGET_PID=$2;  shift 2 ;;
        -w|--window) PROXIMITY_PCT=$2; shift 2 ;;
        -i|--interval) POLL_INTERVAL=$2; shift 2 ;;
        -g|--grace)  GRACE=$2; shift 2 ;;
        -y|--yes)    POWEROFF=1; shift ;;
        --poweroff)   POWEROFF=1; shift ;;
        --no-poweroff) POWEROFF=0; shift ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        --debug-parse) DEBUG_PARSE=1; shift ;;
        --) shift; CMD=("$@"); break ;;
        *)           die "opção desconhecida: $1" ;;
    esac
done

if [ "$DEBUG_PARSE" -eq 1 ]; then
    printf 'EST_SECS=%s (%dh%02dm)\n' "$EST_SECS" $((EST_SECS/3600)) $(( (EST_SECS%3600)/60 ))
    exit 0
fi

# ── decidir o shutdown ANTES (com o usuário presente — nada de perguntar de madrugada)
if [ -z "$POWEROFF" ]; then
    if [ -t 0 ]; then
        read -r -p "[night_sim] Desligar o PC quando a simulação terminar? [S/n] " ans
        case "$ans" in
            n|N) POWEROFF=0 ;;
            *)   POWEROFF=1 ;;
        esac
    else
        # sem terminal (nohup/pipe): mantém o propósito do script — desliga no
        # fim, a menos que --no-poweroff tenha sido passado.
        POWEROFF=1
        log "stdin não é um terminal — shutdown ao fim ATIVADO (padrão; use --no-poweroff para desativar)."
    fi
fi
if [ "$POWEROFF" -eq 1 ]; then
    log "Shutdown ao fim: ATIVADO."
else
    log "Shutdown ao fim: DESATIVADO — o PC ficará ligado quando terminar."
fi

# ── resolver o processo alvo ────────────────────────────────────────────────────────────────────────
START_TS=$(date +%s)   # âncora ANTES do spawn — sincroniza o relógio com o início real do processo

if [ -n "$TARGET_PID" ]; then
    kill -0 "$TARGET_PID" 2>/dev/null || die "PID $TARGET_PID não existe"
elif [ ${#CMD[@]} -gt 0 ]; then
    log "Iniciando: ${CMD[*]}"
    nohup "${CMD[@]}" > /tmp/qga_sim.log 2>&1 &
    TARGET_PID=$!
    log "PID da simulação: $TARGET_PID  (log: /tmp/qga_sim.log)"

    # checagem imediata: falha nos primeiros segundos (typo, python ausente) detectada em 5s,
    # não no primeiro chunk de 10 min.
    sleep 5
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        log "AVISO: o comando terminou nos primeiros segundos — provável erro de execução."
        log "Veja /tmp/qga_sim.log  →  NÃO vou desligar o PC."
        exit 0
    fi
    log "Simulação iniciada com sucesso."
else
    # autodetectar qga.py
    TARGET_PID=$(pgrep -f 'python.*qga\.py' | head -1)
    [ -n "$TARGET_PID" ] || die "nenhum processo qga.py encontrado — informe o PID (-p) ou um comando"
    log "Detectado qga.py com PID $TARGET_PID"
fi

if [ "$EST_SECS" -ge 3600 ]; then
    HUMAN_T=$(printf '%dh%02dm' $((EST_SECS/3600)) $(( (EST_SECS%3600)/60 )))
else
    HUMAN_T=$(printf '%dm%02ds' $((EST_SECS/60)) $(( EST_SECS%60 )))
fi
DEADLINE_TS=$(( START_TS + EST_SECS ))
POLL_WINDOW_TS=$(( START_TS + EST_SECS * (100 - PROXIMITY_PCT) / 100 ))
PROX_MIN=$(( (DEADLINE_TS - POLL_WINDOW_TS) / 60 ))

log "Processo: PID $TARGET_PID"
log "Duração estimada: $HUMAN_T  (pollagem ativa nos últimos $PROX_MIN min)"
log "Início real:      $(date '+%d/%m %H:%M')"
log "NOTA: estimativa generosa demais = procedência de crash falsa. Prefira estimar por baixo."

# ── Fase 1: dorme quase sem consumir até a janela de proximidade ───────────────────────────────────
CHUNK=600   # 10 min por sono — rise de wakeup mínimo, consumo ~zero
while :; do
    NOW=$(date +%s)
    REMAIN_WINDOW=$(( POLL_WINDOW_TS - NOW ))

    if [ "$REMAIN_WINDOW" -le 0 ]; then
        break
    fi

    SLEEP_N=$CHUNK
    [ "$REMAIN_WINDOW" -lt "$CHUNK" ] && SLEEP_N=$REMAIN_WINDOW
    sleep "$SLEEP_N"

    # poll leve de segurança: se o processo sumiu MUITO antes do estimado,
    # provavelmente travou/terminou cedo — alerta e NÃO desliga.
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        ELAPSED=$(( $(date +%s) - START_TS ))
        log "AVISO: processo morreu após $((ELAPSED/60)) min (estimativa era $HUMAN_T)."
        log "Provável crash — NÃO vou desligar o PC. Verifique /tmp/qga_sim.log"
        exit 0
    fi

    ELAPSED=$(( $(date +%s) - START_TS ))
    log "Fase 1 — decorrido $((ELAPSED/60)) min de $HUMAN_T estimados; processo OK."
done

# ── Fase 2 — janela de proximidade: pollagem ativa ────────────────────────────────────────────────
log "Entrando na fase 2: pollando a cada ${POLL_INTERVAL}s até o fim."

while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep "$POLL_INTERVAL"
    ELAPSED=$(( $(date +%s) - START_TS ))
    log "não terminou ainda ($((ELAPSED/60)) min decorridos)"
done

# ── fim detectado: confirmar e desligar ────────────────────────────────────────────────────────────
log "Simulação terminou. Esperando ${GRACE}s (flush de CSV/processos)..."
sleep "$GRACE"

if kill -0 "$TARGET_PID" 2>/dev/null; then
    log "AVISO: processo voltou a existir (PID reutilizado?). Cancelando shutdown."
    exit 0
fi

log "Confirmado: PID $TARGET_PID não existe mais."

if [ "$POWEROFF" -ne 1 ]; then
    log "Shutdown desativado (decisão do início) — monitor encerrando sem desligar."
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    log "(dry-run) Desligaria agora."
    exit 0
fi

# desligar de verdade: tenta primeiro SEM sudo (sessão desktop via polkit costuma permitir),
# depois sudo sem senha (NOPASSWD no sudoers), depois sudo normal (pede senha — só funciona
# se o monitor estiver em primeiro plano num terminal).
if systemctl poweroff 2>/dev/null; then
    log "Desligando agora (sem sudo)."
elif sudo -n systemctl poweroff 2>/dev/null; then
    log "Desligando agora (sudo sem senha)."
elif command -v shutdown >/dev/null 2>&1; then
    log "Executando: sudo shutdown -h now  (depende de NOPASSWD; sudo não pode pedir senha)"
    sudo shutdown -h now 2>/dev/null || log "ERRO: shutdown falhou — o monitor detached não pode digitar a senha de sudo."
else
    log "ERRO: nem systemctl nem shutdown disponíveis."
    exit 1
fi
