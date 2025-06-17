import pickle
import os
import json
from pprint import pprint

# 1. 加载pkl文件
try:
    # 替换为你的实际pkl文件路径
    pkl_path = r'C:/Users/10133/Desktop/drl-alns/dln-alns/DR-ALNS/code/src/routing/cvrp/data/cvrp_20_10000.pkl'

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"成功加载pkl文件: {type(data)}")

except Exception as e:
    print(f"加载失败: {str(e)}")
    exit()

# 2. 准备桌面路径
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
output_path = os.path.join(desktop, 'converted_data.txt')

# 3. 根据数据类型转换并保存
try:
    with open(output_path, 'w', encoding='utf-8') as txt_file:
        # 情况1: 数据是字典或列表（使用JSON格式）
        if isinstance(data, (dict, list)):
            txt_file.write("==== JSON 格式 ====\n")
            json.dump(data, txt_file, indent=4, ensure_ascii=False)

        # 情况2: 数据是字符串
        elif isinstance(data, str):
            txt_file.write("==== 纯文本 ====\n")
            txt_file.write(data)

        # 情况3: 数据是Pandas DataFrame
        elif 'pandas' in str(type(data)):
            txt_file.write("==== DataFrame 结构 ====\n")
            txt_file.write(str(data))

        # 情况4: 其他类型（使用pprint美化输出）
        else:
            txt_file.write("==== 格式化输出 ====\n")
            pprint(data, stream=txt_file, width=100, depth=3)

    print(f"✅ 转换成功！文件已保存到桌面: {output_path}")

except Exception as e:
    print(f"保存失败: {str(e)}")