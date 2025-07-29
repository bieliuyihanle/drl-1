import pandas as pd
from pathlib import Path
import helper_functions
from run_drl_alns_cvrp import run_algo, PARAMETERS_FILE, DEFAULT_RESULTS_ROOT


def batch_run(
    param_file: str = PARAMETERS_FILE,
    instance_folder: str | None = None,
    runs: int = 10,
    results_path: str = "batch_results.xlsx",
) -> None:
    """Run DRL-ALNS on all CVRP instances multiple times and save the results."""
    params = helper_functions.readJSONFile(param_file)

    base_path = Path(__file__).parent
    if instance_folder is None:
        instance_folder = base_path / "data"
    instance_folder = Path(instance_folder)

    records = []
    for inst_file in sorted(instance_folder.glob("*.txt")):
        params["instance_file"] = str(Path("routing/cvrp/data") / inst_file.name)
        for seed in range(runs):
            params["rseed"] = seed
            exp_name = f"drl_alns_{inst_file.stem}_seed{seed}"
            _, best_obj = run_algo(DEFAULT_RESULTS_ROOT, exp_name, **params)
            records.append(
                {
                    "instance": inst_file.stem,
                    "seed": seed,
                    "best_objective": best_obj,
                }
            )

    df = pd.DataFrame(records)
    df.to_excel(results_path, index=False)


if __name__ == "__main__":
    batch_run()