import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, 'C:/Users/10133/Desktop/DR-ALNS-master/DR-ALNS/code/src')  # 替换为你的本地项目路径
import pandas as pd

from rl.environments.cvrpAlnsEnv_LSA1 import cvrpAlnsEnv_LSA1
import helper_functions
from routing.cvrp.alns_cvrp import cvrp_helper_functions
from stable_baselines3 import PPO

DEFAULT_RESULTS_ROOT = "single_runs/"
PARAMETERS_FILE = "configs/drl_alns_cvrp_debug.json"

# def run_algo(folder, exp_name, client=None, **kwargs):
#     instance_nr = kwargs['instance_nr']
#     seed = kwargs['rseed']
#     iterations = kwargs['iterations']
#
#     base_path = Path(__file__).parent.parent.parent
#     instance_file = str(base_path.joinpath(kwargs['instance_file']))
#     model_path = base_path / kwargs['model_directory'] / 'model'
#     model = PPO.load(model_path)
#
#     parameters = {'environment': {'iterations': iterations, 'instance_nr': [instance_nr], 'instance_file': instance_file}}
#     env = cvrpAlnsEnv_LSA1(parameters)
#     env.run(model)
#     best_objective = env.best_solution.objective()
#     print("best_obj", best_objective)
#
#     Path(folder).mkdir(parents=True, exist_ok=True)
#     with open(folder + exp_name + ".csv", "w") as f:
#         writer = csv.writer(f)
#         writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective', 'instance_file'])
#         writer.writerow([instance_nr, seed, iterations, env.best_solution.routes, best_objective, kwargs['instance_file']])
#
#     return [], best_objective


# def run_algo(folder, exp_name, client=None, **kwargs):
#     instance_nr = kwargs['instance_nr']
#     seed = kwargs['rseed']
#     iterations = kwargs['iterations']
#
#     base_path = Path(__file__).resolve().parents[2]
#
#     instance_folder = str(base_path.joinpath(kwargs["instance_folder"]))
#     instances = cvrp_helper_functions.list_problem_files(instance_folder)
#
#     model_path = base_path / kwargs['model_directory'] / 'model'
#     model = PPO.load(model_path)
#
#     parameters = {'environment': {'iterations': iterations, 'instance_nr': [instance_nr], 'instance_file': instance_file}}
#     env = cvrpAlnsEnv_LSA1(parameters)
#     env.run(model)
#     best_objective = env.best_solution.objective()
#     print("best_obj", best_objective)
#
#     Path(folder).mkdir(parents=True, exist_ok=True)
#     with open(folder + exp_name + ".csv", "w") as f:
#         writer = csv.writer(f)
#         writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective', 'instance_file'])
#         writer.writerow([instance_nr, seed, iterations, env.best_solution.routes, best_objective, kwargs['instance_file']])
#
#     return [], best_objective
#
#
# def main(param_file=PARAMETERS_FILE):
#     parameters = helper_functions.readJSONFile(param_file)
#
#     folder = DEFAULT_RESULTS_ROOT
#     exp_name = 'drl_alns' + str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])
#
#     best_objective = run_algo(folder, exp_name, **parameters)
#     return best_objective


## 普通
# def run_algo_for_instance(instance_file: str, model, iterations: int, rseed: int):
#     # 从文件路径中提取实例名（不含扩展名）
#     instance_name = Path(instance_file).stem
#
#     parameters = {
#         'environment': {
#             'iterations': iterations,
#             'instance_nr': 100,  # 使用文件名作为ID
#             'instance_file': instance_file,
#             "instance_folder": "routing/cvrp/data"
#         }
#     }
#
#     env = cvrpAlnsEnv_LSA1(parameters)
#     # env.run(model)
#
#     env.run_time_limit(model)
#
#     # objective_history = env.run_time_limit(model)
#
#     best_objective = env.best_solution.objective()
#
#     print(f"  - 处理完成: {instance_name}, 最优目标值: {best_objective}")
#
#     # 将文件名和最优值返回，由主函数统一处理
#     return instance_name, best_objective
#     # return instance_name, best_objective, objective_history

## 正常
# def main(param_file=PARAMETERS_FILE):
#     """
#     (原main函数)
#     负责读取配置、加载模型、遍历所有实例并汇总结果。
#     """
#     # 1. 读取全局参数
#     parameters = helper_functions.readJSONFile(param_file)
#     seed = parameters['rseed']
#     iterations = parameters['iterations']
#
#     # 2. 构建路径。使用 .resolve() 来获取绝对路径，更稳健
#     base_path = Path(__file__).resolve().parent.parent.parent
#
#     # 3. 高效操作：在循环外执行一次
#     # 加载模型（仅一次）
#     model_path = base_path / parameters['model_directory'] / 'model'
#     model = PPO.load(model_path)
#
#     # 获取所有实例文件列表（仅一次）
#     instance_folder_path = str(base_path.joinpath(parameters["instance_folder"]))
#     all_instance_files = cvrp_helper_functions.list_problem_files(instance_folder_path)
#
#     if not all_instance_files:
#         print(f"\n错误：在文件夹 '{instance_folder_path}' 中没有找到任何问题文件。请检查路径。")
#         return
#
#     print(f"成功找到 {len(all_instance_files)} 个实例文件。开始处理...")
#
#     all_aggregated_results = []
#     # 4. 遍历文件列表，对每个文件调用“处理函数”
#     for i, instance_file_path in enumerate(all_instance_files):
#         # print(instance_file_path)
#         print(f"\n正在处理第 {i + 1}/{len(all_instance_files)} 个实例...")
#         instance_objectives = []  # 存储当前实例10次运行的所有目标值
#         instance_name = ""
#         # 调用函数处理单个实例
#         for i in range(10):
#             temp_instance_name, best_objective = run_algo_for_instance(
#                 instance_file=instance_file_path,
#                 model=model,  # 传入已加载的模型
#                 iterations=iterations,
#                 rseed=seed
#             )
#             if not instance_name:  # 仅在第一次运行时获取实例名称
#                 instance_name = temp_instance_name
#
#             instance_objectives.append(best_objective)
#         mean_objective = np.mean(instance_objectives)
#         std_dev_objective = np.std(instance_objectives)
#
#         print(f"  -> 实例处理完成: 平均目标值 = {mean_objective:.2f}, 标准差 = {std_dev_objective:.2f}")
#
#         # 收集汇总结果
#         all_aggregated_results.append({
#             'instance_name': instance_name,
#             'mean_objective': mean_objective,
#             'std_dev_objective': std_dev_objective
#         })
#
#         # 6. 所有实例处理完毕后，将所有结果一次性写入单个CSV文件
#     if not all_aggregated_results:
#         print("\n没有可供保存的结果。")
#         return
#
#     output_folder = Path(DEFAULT_RESULTS_ROOT)
#     output_folder.mkdir(parents=True, exist_ok=True)
#     # 为汇总文件定义一个更具描述性的文件名
#     summary_file_path = output_folder / "summary_all_instances_aggregated 1s and 30thres.csv"
#
#     print(f"\n全部处理完成！正在将 {len(all_aggregated_results)} 条汇总结果保存到: {summary_file_path}")
#
#     with open(summary_file_path, "w", newline='', encoding='utf-8') as f:
#         # 定义新的CSV文件列名
#         fieldnames = ['instance_name', 'mean_objective', 'std_dev_objective']
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#
#         writer.writeheader()
#         for result in all_aggregated_results:
#             # 将全局参数和汇总结果合并写入一行
#             writer.writerow({
#                 'instance_name': result['instance_name'],
#                 'mean_objective': f"{result['mean_objective']:.4f}",  # 格式化浮点数输出
#                 'std_dev_objective': f"{result['std_dev_objective']:.4f}",
#             })
#
#     print("成功保存！程序结束。")

# 单次
# def main(param_file=PARAMETERS_FILE):
#     """
#     (原main函数)
#     负责读取配置、加载模型、遍历所有实例并汇总结果。
#     """
#     # 1. 读取全局参数
#     parameters = helper_functions.readJSONFile(param_file)
#     seed = parameters['rseed']
#     iterations = parameters['iterations']
#
#     # 2. 构建路径。使用 .resolve() 来获取绝对路径，更稳健
#     base_path = Path(__file__).resolve().parent.parent.parent
#
#     # 3. 高效操作：在循环外执行一次
#     # 加载模型（仅一次）
#     model_path = base_path / parameters['model_directory'] / 'model'
#     model = PPO.load(model_path)
#
#     # 获取所有实例文件列表（仅一次）
#     instance_folder_path = str(base_path.joinpath(parameters["instance_folder"]))
#     all_instance_files = cvrp_helper_functions.list_problem_files(instance_folder_path)
#
#     if not all_instance_files:
#         print(f"\n错误：在文件夹 '{instance_folder_path}' 中没有找到任何问题文件。请检查路径。")
#         return
#
#     print(f"成功找到 {len(all_instance_files)} 个实例文件。开始处理...")
#
#     all_aggregated_results = []
#     # 4. 遍历文件列表，对每个文件调用“处理函数”
#     for i, instance_file_path in enumerate(all_instance_files):
#         # print(instance_file_path)
#         print(f"\n正在处理第 {i + 1}/{len(all_instance_files)} 个实例...")
#         instance_objectives = []  # 存储当前实例10次运行的所有目标值
#         instance_name = ""
#         # 调用函数处理单个实例
#         for i in range(10):
#             temp_instance_name, best_objective = run_algo_for_instance(
#                 instance_file=instance_file_path,
#                 model=model,  # 传入已加载的模型
#                 iterations=iterations,
#                 rseed=seed
#             )
#             if not instance_name:  # 仅在第一次运行时获取实例名称
#                 instance_name = temp_instance_name
#
#             instance_objectives.append(best_objective)
#         mean_objective = np.mean(instance_objectives)
#         std_dev_objective = np.std(instance_objectives)
#
#         print(f"  -> 实例处理完成: 平均目标值 = {mean_objective:.2f}, 标准差 = {std_dev_objective:.2f}")
#
#         # 收集汇总结果
#         all_aggregated_results.append({
#             'instance_name': instance_name,
#             'mean_objective': mean_objective,
#             'std_dev_objective': std_dev_objective
#         })
#
#         # 6. 所有实例处理完毕后，将所有结果一次性写入单个CSV文件
#     if not all_aggregated_results:
#         print("\n没有可供保存的结果。")
#         return
#
#     output_folder = Path(DEFAULT_RESULTS_ROOT)
#     output_folder.mkdir(parents=True, exist_ok=True)
#     # 为汇总文件定义一个更具描述性的文件名
#     summary_file_path = output_folder / "summary_all_instances_aggregated.csv"
#
#     print(f"\n全部处理完成！正在将 {len(all_aggregated_results)} 条汇总结果保存到: {summary_file_path}")
#
#     with open(summary_file_path, "w", newline='', encoding='utf-8') as f:
#         # 定义新的CSV文件列名
#         fieldnames = ['instance_name', 'mean_objective', 'std_dev_objective']
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#
#         writer.writeheader()
#         for result in all_aggregated_results:
#             # 将全局参数和汇总结果合并写入一行
#             writer.writerow({
#                 'instance_name': result['instance_name'],
#                 'mean_objective': f"{result['mean_objective']:.4f}",  # 格式化浮点数输出
#                 'std_dev_objective': f"{result['std_dev_objective']:.4f}",
#             })
#
#     print("成功保存！程序结束。")


def plot_and_save_convergence(history, instance_name, output_folder):
    """
    根据(时间, 目标值)历史数据绘制收敛曲线并保存为PNG文件。
    """
    if not history or len(history) < 2:
        print(f"  - 历史数据不足，无法为 {instance_name} 绘制曲线。")
        return

    plt.figure(figsize=(12, 7))

    # --- 核心修改在这里 ---
    # 1. 解包历史数据
    # history 是一个 [(t1, obj1), (t2, obj2), ...] 格式的列表
    # zip(*history) 会将其转换为 ([t1, t2, ...], [obj1, obj2, ...])
    time_points, objective_values = zip(*history)

    # 2. 使用新的数据进行绘图
    plt.plot(time_points, objective_values, linestyle='-', label='Best Objective')
    # --- 修改结束 ---

    # 添加标题和坐标轴标签 (修改xlabel)
    plt.title(f'Convergence Curve for {instance_name}', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)  # <--- X轴标签已更新
    plt.ylabel('Objective Value', fontsize=12)

    # 添加网格和图例
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # 设置X轴的范围，从0开始
    plt.xlim(left=0)

    # 确保输出文件夹存在
    plot_output_folder = output_folder / "convergence_plots"
    plot_output_folder.mkdir(parents=True, exist_ok=True)

    # 保存图像
    clean_instance_name = Path(instance_name).stem
    save_path = plot_output_folder / f"{clean_instance_name}_convergence.png"
    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"  - 收敛曲线已保存到: {save_path}")

def run_algo_for_instance(instance_file: str, model, iterations: int, rseed: int):
    # 从文件路径中提取实例名（不含扩展名）
    instance_name = Path(instance_file).stem

    parameters = {
        'environment': {
            'iterations': iterations,
            'instance_nr': 100,  # 使用文件名作为ID
            'instance_file': instance_file,
            "instance_folder": "routing/cvrp/data"
        }
    }

    env = cvrpAlnsEnv_LSA1(parameters)

    objective_history = env.run_time_limit(model)

    best_objective = env.best_solution.objective()

    print(f"  - 处理完成: {instance_name}, 最优目标值: {best_objective}")

    return instance_name, best_objective, objective_history

# # 收敛曲线
# def main(param_file=PARAMETERS_FILE):
#     """
#     (原main函数)
#     负责读取配置、加载模型、遍历所有实例并汇总结果。
#     """
#     # 1. 读取全局参数
#     parameters = helper_functions.readJSONFile(param_file)
#     seed = parameters['rseed']
#     iterations = parameters['iterations']
#
#     # 2. 构建路径。使用 .resolve() 来获取绝对路径，更稳健
#     base_path = Path(__file__).resolve().parent.parent.parent
#
#     # 3. 高效操作：在循环外执行一次
#     # 加载模型（仅一次）
#     model_path = base_path / parameters['model_directory'] / 'model'
#     model = PPO.load(model_path)
#
#     # 获取所有实例文件列表（仅一次）
#     instance_folder_path = str(base_path.joinpath(parameters["instance_folder"]))
#     all_instance_files = cvrp_helper_functions.list_problem_files(instance_folder_path)
#
#     print(f"找到 {len(all_instance_files)} 个实例。将为每个实例运行 10 秒。")
#
#     # 3. 准备结果收集和输出路径
#     all_results = []
#     output_folder = Path(DEFAULT_RESULTS_ROOT)
#
#     # 4. 循环处理每个实例
#     for i, instance_file_path in enumerate(all_instance_files):
#         instance_name_with_ext = Path(instance_file_path).name
#         print(f"\n[{i+1}/{len(all_instance_files)}] 正在处理: {instance_name_with_ext}")
#
#         instance_name, best_objective, objective_history = run_algo_for_instance(
#             instance_file=instance_file_path,
#             model=model,  # 传入已加载的模型
#             iterations=iterations,
#             rseed=seed
#         )
#
#         # 收集结果
#         all_results.append({
#             'instance_name': instance_name,
#             'best_objective': best_objective
#         })
#
#         if objective_history:
#             plot_and_save_convergence(objective_history, instance_name, output_folder)
#
#         # 5. 所有实例处理完毕后，写入最终的汇总CSV报告
#         summary_file_path = output_folder / "summary_all_instances.csv"
#         print(f"\n全部处理完成！正在将汇总结果保存到: {summary_file_path}")
#
#         with open(summary_file_path, "w", newline='') as f:
#             # 更新列名以反映实验是基于时间的
#             fieldnames = ['instance_name', 'best_objective']
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#
#             writer.writeheader()
#             for result in all_results:
#                 writer.writerow({
#                     'instance_name': result['instance_name'],
#                     'best_objective': result['best_objective'],
#                 })
#
#         print("成功保存！程序结束。")


def main(param_file=PARAMETERS_FILE):
    """
    --- 这是修改后的新主函数 ---
    目标：为每个实例运行DRL-ALNS算法10次，
    并将所有详细的收敛数据保存到一个标准化的CSV文件中。
    """
    # 1. 读取参数，加载模型，获取实例文件列表 (与您原代码类似)
    parameters = helper_functions.readJSONFile(param_file)
    seed = parameters['rseed']
    iterations = parameters['iterations']
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / parameters['model_directory'] / 'model'
    model = PPO.load(model_path)
    instance_folder_path = str(base_path.joinpath(parameters["instance_folder"]))
    all_instance_files = cvrp_helper_functions.list_problem_files(instance_folder_path)

    print(f"模型已加载。找到 {len(all_instance_files)} 个实例。")

    # 2. 准备一个列表，用于收集所有运行的详细数据
    all_convergence_runs = []
    num_runs = 10  # 定义每个实例的运行次数

    # 3. 外层循环：遍历每个实例文件
    for i, instance_file_path in enumerate(all_instance_files):
        instance_name_with_ext = Path(instance_file_path).name
        print(f"\n[{i+1}/{len(all_instance_files)}] 正在处理实例: {instance_name_with_ext}")

        # 4. 内层循环：对当前实例重复运行 `num_runs` 次
        for run_num in range(1, num_runs + 1):
            print(f"  - 第 {run_num}/{num_runs} 次运行...")

            # 为当前实例运行一次算法
            _, best_objective, objective_history = run_algo_for_instance(
                instance_file=instance_file_path,
                model=model,
                iterations=iterations,
                rseed=seed + run_num  # 每次运行使用不同的随机种子以保证结果差异
            )

            # 5. 解包收敛数据并以标准格式记录
            if objective_history:
                times, costs = zip(*objective_history)
            else:
                times, costs = [], []

            all_convergence_runs.append({
                "File": instance_name_with_ext,
                "Algorithm": "DRL-ALNS",
                "Run": run_num,
                "Final_Cost": best_objective,
                "Costs": list(costs),  # 将元组转换为列表
                "Times": list(times)   # 将元组转换为列表
            })

    # 6. 所有实验结束后，将收集到的所有数据一次性保存到CSV文件
    print("\n所有实验运行完毕。正在保存详细收敛数据...")
    output_folder = Path(DEFAULT_RESULTS_ROOT)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "drl_alns_convergence_runs.csv"

    convergence_df = pd.DataFrame(all_convergence_runs)
    convergence_df.to_csv(output_file, index=False)

    print(f"数据已成功保存至: {output_file}")

if __name__ == "__main__":
    main()