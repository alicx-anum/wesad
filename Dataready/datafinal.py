import os
import pandas as pd

# 获取当前目录下的所有CSV文件
csv_files = [f for f in os.listdir('.') if f.endswith('_ECG_data.csv')]

# 创建一个空的列表来存储所有受试者的数据
all_subjects_data = []

# 遍历每个CSV文件
for file in csv_files:
    print(f"Loading data from {file}")
    # 加载每个CSV文件的数据
    subject_data = pd.read_csv(file)
    all_subjects_data.append(subject_data)

# 合并所有受试者的数据
combined_data = pd.concat(all_subjects_data, ignore_index=True)

# 保存合并后的数据为一个 CSV 文件
combined_data.to_csv('wesad_data.csv', index=False)
print("All subject data has been combined into 'wesad_data.csv'")
