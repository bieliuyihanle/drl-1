import csv
from pathlib import Path
import sys
sys.path.insert(0, 'C:/Users/10133/Desktop/DR-ALNS-master/DR-ALNS-master/code/src')  # 替换为你的本地项目路径
from rl.environments.cvrpAlnsEnv_LSA1 import cvrpAlnsEnv_LSA1
import helper_functions

from stable_baselines3 import PPO

DEFAULT_RESULTS_ROOT = "single_runs/"
PARAMETERS_FILE = "configs/drl_alns_cvrp_debug.json"


def run_algo(folder, exp_name, client=None, **kwargs):
    seed = kwargs["rseed"]
    iterations = kwargs["iterations"]

    base_path = Path(__file__).parent.parent.parent
    instance_file = str(base_path.joinpath(kwargs["instance_file"]))
    model_path = base_path / kwargs["model_directory"] / "model"
    model = PPO.load(model_path)

    parameters = {
        "environment": {
            "iterations": iterations,
            "instance_folder": str(Path(instance_file).parent),
        }
    }
    env = cvrpAlnsEnv_LSA1(parameters)
    env.instances = [instance_file]
    env.instance = instance_file
    env.run(model)
    best_objective = env.best_solution.objective()

    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(folder + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instance",
                "rseed",
                "iterations",
                "solution",
                "best_objective",
            ]
        )
        writer.writerow(
            [
                Path(instance_file).name,
                seed,
                iterations,
                env.best_solution.routes,
                best_objective,
            ]
        )
    return [], best_objective

def main(param_file=PARAMETERS_FILE):
    parameters = helper_functions.readJSONFile(param_file)

    folder = DEFAULT_RESULTS_ROOT

    instance = Path(parameters["instance_file"]).stem
    exp_name = f"drl_alns_{instance}_{parameters['rseed']}"

    best_objective = run_algo(folder, exp_name, **parameters)
    return best_objective


if __name__ == "__main__":
    main()