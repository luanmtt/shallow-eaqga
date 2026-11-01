"""Prints herdados do código-base QGA (qga.py e ParallelQuantumGA.py).

Referência dos formatos de saída das simulações **batch** e **single run**,
preservados para comparação durante a reescrita do `eaqga`. Não é a API
pública do pacote: usa apenas os utilitários de terminal de `eaqga.utils`.

Os textos, cores e alinhamentos são os do repositório antigo; qualquer
divergência aqui é bug deste arquivo, não uma escolha de design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from eaqga.utils.embellish import sepComment, stylePrint
from eaqga.utils.formats import formatMantissa
from eaqga.utils.progressbar import ProgressBar

__all__ = [
    # single run
    "format_batch_bar",
    "format_log_line",
    "format_run_time",
    "make_progress_bars",
    "print_advanced_eda_prompt",
    "print_advanced_eda_saved",
    "print_animations_prompt",
    "print_animations_saved",
    "print_circuit_saved",
    "print_colorscheme_prompt",
    "print_export_prompt",
    "print_export_paths",
    "print_generating_2d",
    "print_generating_3d",
    "print_graphs_prompt",
    "print_graphs_saved",
    "print_run_complete",
    "print_run_header",
    "print_run_legend",
    # batch
    "estimate_storage_per_run",
    "format_duration",
    "format_sim_failed",
    "format_sim_interrupted",
    "format_sim_line",
    "format_storage",
    "print_batch_cancelled",
    "print_batch_complete",
    "print_batch_confirm_prompt",
    "print_batch_failures",
    "print_batch_legend",
    "print_batch_menu",
    "print_batch_no_terminal",
    "print_batch_starting",
    "print_config_saved",
    "print_config_warning",
    "print_final_location",
    "print_gate_failures",
    "print_gate_set_header",
    "print_gate_stats",
    "print_generating_graphs",
    "print_graphs_warning",
    "print_multirun_animations",
    "print_multirun_warning",
]

C = "#f5c2e7"
WARN = "#f3af41"
ERROR = "#ff6b6b"
MUTED = "#666666"
RULE = "#444444"
CLASSICAL = "#e74c3c"
QUANTUM = "#3498db"
BATCH = "#dcd7ba"


def _dot() -> str:
    """Marcador ◦ colorido (calculado a cada chamada: o gate de cor muda)."""
    return stylePrint("◦", C)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single run (não batch) — ParallelQuantumGA.run()


def make_progress_bars() -> tuple[ProgressBar, ProgressBar, ProgressBar]:
    """As três barras do single run: geração, fitness e batch."""
    gen_bar = ProgressBar(total=1, style="keras", fg_color="#CC3363", bg_color="#5b5f66")
    fit_bar = ProgressBar(total=1, style="keras", fg_color="#f6cdb8", bg_color="#5b5f66")
    batch_bar = ProgressBar(total=1, style="keras", fg_color="#dcd7ba", bg_color="#5b5f66")
    return gen_bar, fit_bar, batch_bar


def print_run_header(
    *,
    fitness_name: str,
    minimize: bool,
    selection: str,
    crossover: str,
    population_size: int,
    generations: int,
    target_fitness: float,
    door_type_name: str,
    exp_retrieval: Sequence[object],
) -> None:
    """Cabeçalho exibido no início de uma avaliação do QGA."""
    dot = _dot()
    header = (
        f"\n{sepComment(40, stylePrint('Iniciando availiação QGA', C, bold=True))}\n"

        f"  {dot} Fitness Function {stylePrint(fitness_name, C, bold=True)}, minimize = {stylePrint(str(minimize), C, bold=True)}\n"
        f"  {dot} Selection: {stylePrint(selection, C, bold=True)}\n"
        f"  {dot} Crossover: {stylePrint(crossover, C, bold=True)}\n"
        f"  {dot} Population: {stylePrint(str(population_size), C, bold=True)}, Generation Lifespan: {stylePrint(str(generations), C, bold=True)}\n"
        f"  {dot} Best possible fitness: {stylePrint(str(target_fitness), C, bold=True)}\n"
        f"  {dot} Door Type: {stylePrint(door_type_name, C, bold=True)}\n"
        f"  {dot} exp_retrieval: {stylePrint(str(exp_retrieval[0]), C, bold=True)} every {stylePrint(str(exp_retrieval[1]), C, bold=True)} generations\n"
    )
    print(header, flush=True)


def print_run_legend() -> None:
    """Legenda das colunas da barra de progresso do single run."""
    legend = (
        "                   generation bar         fitness bar           "
        "   gen/s      current best     avg\n"
    )
    print(legend, flush=True)


def format_log_line(
    *,
    generation: int,
    generations: int,
    elapsed: float,
    best_fitness: float,
    avg: float,
    gen_bar: ProgressBar,
    fit_bar: ProgressBar,
    fitness_ratio: float,
) -> str:
    """Linha de progresso por geração (`_format_log_line` do código antigo)."""
    current = generation + 1
    total = generations
    total_width = len(str(total))

    gen_ratio = current / total if total else 0
    gen_bar_str = gen_bar.bar(gen_ratio)
    fit_bar_str = fit_bar.bar(fitness_ratio)

    gen_s = current / elapsed if elapsed > 0 else 0.0
    gen_label = stylePrint(f"Generation {current:>{total_width}}/{total}", C, bold=True)

    # Acima de 1, float normal; abaixo, notação de mantissa.
    if abs(best_fitness) > 1:
        fit_str = f"{best_fitness:>12.4f}"
    else:
        fit_str = formatMantissa(best_fitness)

    return (
        f"{gen_label}   {gen_bar_str}   {fit_bar_str}   "
        f"{gen_s:>6.2f}g/s      {fit_str}      {avg:.4f}"
    )


def format_batch_bar(*, current: int, total: int, batch_bar: ProgressBar) -> str:
    """Barra de batch exibida à esquerda da linha de geração."""
    total_width = len(str(total))
    ratio = current / total if total else 0
    label = stylePrint(f"Batch {current:>{total_width}}/{total}", BATCH, bold=True)
    return f"{label}   {batch_bar.bar(ratio)}"


def format_run_time(seconds: float) -> str:
    """Tempo do single run, sem '0min'/'0sec'."""
    minutes = int(seconds // 60)
    remainder = seconds % 60

    if minutes > 0 and remainder > 0:
        return f"{minutes}min {remainder:.1f}sec"
    if minutes > 0:
        return f"{minutes}min"
    return f"{remainder:.1f}sec"


def print_run_complete(
    *,
    elapsed: float,
    best_fitness: float,
    best_location: object | None,
) -> None:
    """Bloco 'Run Complete' do single run."""
    dot = _dot()
    print(f"\n{sepComment(40, stylePrint('Run Complete', C, bold=True))}")
    print(f"  {dot} Simulation time: {stylePrint(format_run_time(elapsed), C, bold=True)}")
    print(f"  {dot} Best Fitness: {stylePrint(formatMantissa(best_fitness), C, bold=True)}")
    if best_location is not None:
        print(f"  {dot} Best Location: {stylePrint(str(best_location), C)}")


def print_export_prompt() -> None:
    print("\n  • Export data to CSV? [y/n]")


def print_export_paths(evolution_path: Path, metrics_path: Path) -> None:
    dot = _dot()
    print(f"  {dot} Evolution CSV: {stylePrint(str(evolution_path), C)}")
    print(f"  {dot} Metrics CSV: {stylePrint(str(metrics_path), C)}")


def print_circuit_saved(circuit_path: Path) -> None:
    print(f"  {_dot()} Circuit saved to: {stylePrint(str(circuit_path), C)}")


def print_graphs_prompt() -> None:
    print("\n  • Generate graphs? [y/n]")


def print_colorscheme_prompt() -> None:
    print("\n  • What is the desired colorscheme? (p for paper / v for visualization)")


def print_graphs_saved(output_dir: Path) -> None:
    print(f"  {_dot()} Graphs saved to: {stylePrint(str(output_dir), C)}")


def print_advanced_eda_prompt() -> None:
    print("\n  • Generate advanced diversity EDA? [y/n]")


def print_advanced_eda_saved(output_dir: Path) -> None:
    print(f"  {_dot()} Advanced EDA saved to: {stylePrint(str(output_dir), C)}")


def print_animations_prompt() -> None:
    print("\n  • Generate animations? [y/n]")


def print_generating_2d() -> None:
    print(f"  {_dot()} Generating 2D animation...")


def print_generating_3d() -> None:
    print(f"  {_dot()} Generating 3D animation...")


def print_animations_saved(output_dir: Path) -> None:
    print(f"  {_dot()} Animations saved to: {stylePrint(str(output_dir), C)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch — qga.py


def format_duration(seconds: float) -> str:
    """Formata segundos em dias/horas/minutos/segundos, sem valores zero."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    if secs > 0 or not parts:
        parts.append(f"{secs:.0f}sec")

    return " ".join(parts)


def estimate_storage_per_run() -> float:
    """Estimativa fixa de armazenamento por run, em KB."""
    return 56.0


def format_storage(total_kb: float) -> str:
    """Armazenamento total em GB/MB/KB, conforme a magnitude."""
    if total_kb >= 1024 * 1024:
        return f"{total_kb / (1024 * 1024):.1f} GB"
    if total_kb >= 1024:
        return f"{total_kb / 1024:.1f} MB"
    return f"{total_kb:.0f} KB"


def print_batch_menu(
    *,
    gate_sets: Iterable[str],
    n_sims: int,
    total_sims: int,
    total_time_est: float,
    avg_time_per_run: float,
    storage_per_run: float,
    total_storage_kb: float,
) -> None:
    """Menu de confirmação do batch, antes de começar."""
    dot = _dot()
    print()
    print(sepComment(50, stylePrint("Batch Simulation", C, bold=True)))
    print(f"  {dot} Gate sets: {stylePrint(', '.join(gate_sets), C, bold=True)}")
    print(f"  {dot} Simulations per set: {stylePrint(str(n_sims), C, bold=True)}")
    print(f"  {dot} Total simulations: {stylePrint(str(total_sims), C, bold=True)}")
    print(f"  {dot} Estimated time: {stylePrint(format_duration(total_time_est), C, bold=True)} (avg {avg_time_per_run:.0f}s per run)")
    print(f"  {dot} Estimated storage: {stylePrint(format_storage(total_storage_kb), C, bold=True)} (avg {storage_per_run:.0f}KB per run)")
    print()


def print_batch_confirm_prompt() -> None:
    print(f"  {stylePrint('Are you sure you want to continue? [y/n]', WARN)}")


def print_batch_cancelled() -> None:
    print(f"\n  {_dot()} Cancelled.\n")


def print_batch_no_terminal() -> None:
    print(f"  {stylePrint('(sem terminal — confirmação automática)', MUTED)}")


def print_batch_starting() -> None:
    print(f"\n  {_dot()} Starting batch simulation...\n")


def print_batch_legend() -> None:
    print(f"  {stylePrint('Legend:', MUTED)} gen/s {stylePrint('|', MUTED)} current best {stylePrint('|', MUTED)} avg")
    print()


def print_gate_set_header(*, gate_label: str, door_type: int) -> None:
    """Separador e rótulo do gate set em execução."""
    gate_color = CLASSICAL if door_type == 0 else QUANTUM
    print(f"  {stylePrint('─' * 40, RULE)}")
    print(f"  {stylePrint(gate_label.upper(), gate_color, bold=True)} (door_type={door_type})")
    print()


def format_sim_line(
    *,
    fitness_name: str,
    gate_label: str,
    sim: int,
    n_sims: int,
    best_fitness: float,
    sim_time: float,
) -> str:
    """Linha de uma simulação concluída com sucesso."""
    return (
        f"  {_dot()} {fitness_name} - {gate_label} @ run {sim}/{n_sims} — "
        f"best {best_fitness:.6g} ({sim_time:.1f}s)"
    )


def format_sim_failed(
    *,
    fitness_name: str,
    gate_label: str,
    sim: int,
    n_sims: int,
    exc: str,
) -> str:
    """Linha de uma simulação que falhou (isolada, o batch continua)."""
    return f"  {_dot()} {fitness_name} - {gate_label} @ run {sim}/{n_sims} — FALHOU ({exc})"


def format_sim_interrupted(*, sim: int) -> str:
    """Linha exibida no Ctrl+C, antes de salvar o parcial."""
    return f"  {_dot()} interrompido pelo usuário na sim {sim} — salvando parcial..."


def print_gate_stats(
    *,
    gate_label: str,
    fit_mean: float,
    fit_std: float,
    ent_mean: float,
    ent_std: float,
) -> None:
    """Resumo estatístico de um gate set concluído."""
    print()
    print(f"  {_dot()} {gate_label}: Fitness {stylePrint(f'{fit_mean:.4f} ± {fit_std:.4f}', C)}, "
          f"Gate Entropy {stylePrint(f'{ent_mean:.4f} ± {ent_std:.4f}', C)}")
    print()


def print_gate_failures(*, n_fail: int, n_sims: int) -> None:
    print(f"  {_dot()} {stylePrint('Warning', WARN)}: {n_fail}/{n_sims} sims falharam (ver coluna 'error' no CSV)")


def print_config_saved(config_path: Path) -> None:
    print(f"  {_dot()} Config saved to: {stylePrint(str(config_path), C)}")


def print_config_warning(exc: Exception) -> None:
    print(f"  {stylePrint('Warning', WARN)}: Could not save config.json: {exc}")


def print_batch_complete(
    *,
    total_time: float,
    batch_dir: Path,
    csv_paths: Mapping[str, Path],
) -> None:
    """Bloco 'Batch Complete'."""
    dot = _dot()
    print()
    print()
    print(sepComment(50, stylePrint("Batch Complete", C, bold=True)))
    print(f"  {dot} Total time: {stylePrint(format_duration(total_time), C, bold=True)}")
    print(f"  {dot} Results saved to: {stylePrint(str(batch_dir), C)}")
    for label, path in csv_paths.items():
        print(f"      {label}: {stylePrint(str(path), C)}")
    print()


def print_generating_graphs() -> None:
    print(f"  {_dot()} Generating comparative graphs...")


def print_graphs_warning(exc: Exception) -> None:
    print(f"  {stylePrint('Warning', WARN)}: Could not generate graphs: {exc}")


def print_multirun_animations() -> None:
    print(f"  {_dot()} Generated {stylePrint('multi_run_2d.gif', C)} and {stylePrint('multi_run_3d.gif', C)}")


def print_multirun_warning(exc: Exception) -> None:
    print(f"  {stylePrint('Warning', WARN)}: Could not generate multi-run 2D/3D animations: {exc}")


def print_final_location(dest: Path) -> None:
    print(f"  {_dot()} Final location: {stylePrint(str(dest), C)}")


def print_batch_failures(total_failures: int) -> None:
    print(f"  {_dot()} {stylePrint('Warning', WARN)}: {total_failures} sims falharam no batch (ver coluna 'error' nos CSVs)")
