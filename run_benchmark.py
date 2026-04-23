from __future__ import annotations
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
    max_workers: int = 10,
) -> None:
    examples = load_dataset(dataset)

    if mode == "real":
        react = ReActAgent(runtime="real")
        reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime="real")
    else:
        react = ReActAgent(runtime="mock")
        reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime="mock")

    def run_agent(agent, example, agent_type):
        try:
            return agent.run(example), None
        except Exception as e:
            return None, f"{agent_type} qid={example.qid} error: {e}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        # ReAct phase
        react_task = progress.add_task(f"[cyan]ReAct ({len(examples)} samples)", total=len(examples))
        react_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_agent, react, ex, "ReAct"): ex for ex in examples}
            for future in as_completed(futures):
                record, error = future.result()
                if record:
                    react_records.append(record)
                if error:
                    logger.error(error)
                progress.update(react_task, advance=1)

        # Reflexion phase
        reflexion_task = progress.add_task(f"[magenta]Reflexion ({len(examples)} samples)", total=len(examples))
        reflexion_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_agent, reflexion, ex, "Reflexion"): ex for ex in examples}
            for future in as_completed(futures):
                record, error = future.result()
                if record:
                    reflexion_records.append(record)
                if error:
                    logger.error(error)
                progress.update(reflexion_task, advance=1)

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
