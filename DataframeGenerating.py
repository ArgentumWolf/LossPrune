import pandas as pd
import numpy as np
import random


def generate_random():#随机生成数据框
    """
    生成一个符合要求的学生线上答题设置调整数据框。

    返回:
        pd.DataFrame: 包含 time, top_setting, central_setting, bottom_setting 列的数据框。
                      行数在 10 到 30 之间随机。
                      time 递增，增量服从均值为 5 的正态分布。
                      setting 列的值从 [-2, -1, 0, 1, 2] 中随机选取。
    """
    # 1. 确定行数 (10 到 30 之间的随机整数)
    num_rows = random.randint(10, 30)

    # 2. 生成 time 列
    # 初始时间可以设为 0 或一个小的正数
    current_time = 0.0
    time_values = []
    # 设置正态分布的标准差，例如 1.0 或 2.0。这里我们用 1.5
    time_std_dev = 1.5
    for _ in range(num_rows):
        # 生成增量，确保增量为正数，使时间总是增加
        increment = max(0.01, np.random.uniform(0,30)) # loc 是均值，scale 是标准差
        current_time += increment
        time_values.append(current_time)

    # 3. 生成 setting 列
    settings_options = [-2, -1,0, 1, 2]
    
    #settings_options = [-2, -1, 1, 2]
    top_setting = np.random.choice(settings_options, size=num_rows)
    central_setting = np.random.choice(settings_options, size=num_rows)
    bottom_setting = np.random.choice(settings_options, size=num_rows)

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        'time': time_values,
        'top_setting': top_setting,
        'central_setting': central_setting,
        'bottom_setting': bottom_setting
    })

    # 5. （可选）将 setting 列转换为 Categorical 类型以体现其 'factor' 性质
    # 这对于某些分析可能有用，但对于纯粹的数据生成不是必需的
    # for col in ['top_setting', 'central_setting', 'bottom_setting']:
    #     df[col] = pd.Categorical(df[col], categories=settings_options, ordered=True)

    return df

def generate_negative():#生成只有一种的数据框
    # 1. 确定行数 (10 到 30 之间的随机整数)
    num_rows = random.randint(10, 30)

    # 2. 生成 time 列
    # 初始时间可以设为 0 或一个小的正数
    current_time = 0.0
    time_values = []
    # 设置正态分布的标准差，例如 1.0 或 2.0。这里我们用 1.5
    time_std_dev = 1.5
    for _ in range(num_rows):
        # 生成增量，确保增量为正数，使时间总是增加
        increment = max(0.01, np.random.normal(loc=5.0, scale=time_std_dev)) # loc 是均值，scale 是标准差
        current_time += increment
        time_values.append(current_time)

    # 3. 生成 setting 列
    top_settings = [-2, -1,0]
    central_settings = [-2, -1,0]
    bottom_settings = [-2, -1,0]
    top_setting = np.random.choice(top_settings, size=num_rows)
    central_setting = np.random.choice(central_settings, size=num_rows)
    bottom_setting = np.random.choice(bottom_setting, size=num_rows)

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        'time': time_values,
        'top_setting': top_setting,
        'central_setting': central_setting,
        'bottom_setting': bottom_setting
    })

    # 5. （可选）将 setting 列转换为 Categorical 类型以体现其 'factor' 性质
    # 这对于某些分析可能有用，但对于纯粹的数据生成不是必需的
    # for col in ['top_setting', 'central_setting', 'bottom_setting']:
    #     df[col] = pd.Categorical(df[col], categories=settings_options, ordered=True)

    return df

def generate_positive():#生成只有一种的数据框
    
    # 1. 确定行数 (10 到 30 之间的随机整数)
    num_rows = random.randint(10, 30)

    # 2. 生成 time 列
    # 初始时间可以设为 0 或一个小的正数
    current_time = 0.0
    time_values = []
    # 设置正态分布的标准差，例如 1.0 或 2.0。这里我们用 1.5
    time_std_dev = 1.5
    for _ in range(num_rows):
        # 生成增量，确保增量为正数，使时间总是增加
        increment = max(0.01, np.random.normal(loc=5.0, scale=time_std_dev)) # loc 是均值，scale 是标准差
        current_time += increment
        time_values.append(current_time)

    # 3. 生成 setting 列
    top_settings = [2, 1]
    central_settings = [2, 1]
    bottom_settings = [2, 1]
    top_setting = np.random.choice(top_settings, size=num_rows)
    central_setting = np.random.choice(central_settings, size=num_rows)
    bottom_setting = np.random.choice(bottom_settings, size=num_rows)

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        'time': time_values,
        'top_setting': top_setting,
        'central_setting': central_setting,
        'bottom_setting': bottom_setting
    })

    # 5. （可选）将 setting 列转换为 Categorical 类型以体现其 'factor' 性质
    # 这对于某些分析可能有用，但对于纯粹的数据生成不是必需的
    # for col in ['top_setting', 'central_setting', 'bottom_setting']:
    #     df[col] = pd.Categorical(df[col], categories=settings_options, ordered=True)

    return df

def generate_reset(): # 或者直接修改 v2
    """
    生成一个符合新规则的学生线上答题设置调整数据框 (加权值选择)。

    规则:
    - time 递增，增量服从均值为 5 的正态分布。
    - setting 列 (top, central, bottom):
        - 每行只有一个 setting 列非零。
        - 非零列的值从 [-2, -1, 1, 2] 中随机选取，其中 -2 和 2 的选择概率高于 -1 和 1。
          (例如，-2 和 2 的概率各为 0.375，-1 和 1 的概率各为 0.125)
        - 非零列的位置 (top, central, bottom) 与上一行不同。
    - 行数在 10 到 30 之间随机。

    返回:
        pd.DataFrame: 符合新规则 (加权) 的数据框。
    """
    # 1. 确定行数 (10 到 30 之间的随机整数)
    num_rows = random.randint(10, 30)

    # 2. 生成 time 列 (与之前函数相同)
    current_time = 0.0
    time_values = []
    time_std_dev = 1.5 # 可以调整标准差
    for _ in range(num_rows):
        increment = max(0.01, np.random.normal(loc=5.0, scale=time_std_dev))
        current_time += increment
        time_values.append(current_time)

    # 3. 生成 setting 列 (带加权值选择的新逻辑)
    setting_cols = ['top_setting', 'central_setting', 'bottom_setting']
    col_indices = list(range(len(setting_cols))) # 列索引: [0, 1, 2]

    # --- 修改点：定义非零值选项和它们的概率 ---
    non_zero_options = [-2, -1, 1, 2]
    # 定义概率，确保总和为 1。
    # 例如，让 -2 和 2 的概率是 -1 和 1 的 3 倍。
    # 权重: -2: 3, -1: 1, 1: 1, 2: 3  (总权重 = 8)
    # 概率: P(-2)=3/8, P(-1)=1/8, P(1)=1/8, P(2)=3/8
    value_probabilities = [0.375, 0.125, 0.125, 0.375]
    # 您可以根据需要调整这些概率，只要它们加起来等于 1
    # -----------------------------------------

    # 用于存储每列值的列表
    top_values = []
    central_values = []
    bottom_values = []

    # 追踪上一行非零列的索引
    previous_non_zero_col_index = -1

    for i in range(num_rows):
        # a. 确定当前行允许的非零列索引
        allowed_indices = [idx for idx in col_indices if idx != previous_non_zero_col_index]

        # b. 从允许的索引中随机选择一个作为当前行的非零列
        current_non_zero_col_index = random.choice(allowed_indices)

        # c. --- 修改点：使用 np.random.choice 和指定的概率来选择非零值 ---
        current_value = np.random.choice(non_zero_options, p=value_probabilities)
        # ------------------------------------------------------------

        # d. 创建当前行的设置值列表
        row_settings = [0] * len(setting_cols)
        row_settings[current_non_zero_col_index] = current_value

        # e. 将当前行的值分别添加到各列的列表中
        top_values.append(row_settings[0])
        central_values.append(row_settings[1])
        bottom_values.append(row_settings[2])

        # f. 更新 previous_non_zero_col_index 以备下一行使用
        previous_non_zero_col_index = current_non_zero_col_index

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        'time': time_values,
        'top_setting': top_values,
        'central_setting': central_values,
        'bottom_setting': bottom_values
    })

    return df



def generate_onestep():
    """
    生成一个符合新规则的学生线上答题设置调整数据框 (V5)。

    规则:
    - time 递增，增量服从均值为 5 的正态分布。
    - setting 列 (top, central, bottom):
        - 第 1 行：只有一个 setting 列非零，其值从 [-2, -1, 1, 2] 中随机选取，其他列为 0。
        - 第 N 行 (N>1)：相比第 N-1 行，只有一个 setting 列的值发生改变。
            - 改变的列与“导致第 N-1 行产生的改变”所涉及的列不同。
            - 改变后的新值从 [-2, -1, 0, 1, 2] 中随机选取，*并且新值必须与改变前的值不同*。
    - 行数在 10 到 30 之间随机。

    返回:
        pd.DataFrame: 符合新规则 (V5) 的数据框。
    """
    # 1. 确定行数 (10 到 30 之间的随机整数)
    num_rows = random.randint(10, 30)
    if num_rows < 2: # 规则需要至少两行才能完全体现
        num_rows = 2

    # 2. 生成 time 列 (与之前函数相同)
    current_time = 0.0
    time_values = []
    time_std_dev = 1.5 # 可以调整标准差
    for _ in range(num_rows):
        increment = max(0.01, np.random.normal(loc=5.0, scale=time_std_dev))
        current_time += increment
        time_values.append(current_time)

    # 3. 生成 setting 列 (V5 规则)
    setting_cols = ['top_setting', 'central_setting', 'bottom_setting']
    col_indices = list(range(len(setting_cols))) # 列索引: [0, 1, 2]
    all_options = [-2, -1, 0, 1, 2]          # 所有可能的值
    non_zero_options = [-2, -1, 1, 2]        # 非零值选项 (用于第一行)

    # 用于存储最终的列值
    top_values = []
    central_values = []
    bottom_values = []

    # 状态变量
    previous_settings = [0, 0, 0]  # 存储上一行的完整设置 [top, central, bottom]
    last_changed_col_index = -1    # 记录上一次是哪个列被改变了 (索引 0, 1, 或 2)

    for i in range(num_rows):
        current_settings = [0, 0, 0] # 初始化当前行设置

        if i == 0:
            # --- 处理第一行 (与 V4 相同) ---
            col_to_set_index = random.choice(col_indices)
            value_to_set = random.choice(non_zero_options)
            current_settings = [0, 0, 0]
            current_settings[col_to_set_index] = value_to_set
            last_changed_col_index = col_to_set_index
        else:
            # --- 处理后续行 (i > 0) ---
            # a. 确定本行可以改变的列 (不能是上一次改变的列)
            allowed_indices_to_change = [idx for idx in col_indices if idx != last_changed_col_index]
            # b. 从允许的列中随机选择一个进行改变
            col_to_change_index = random.choice(allowed_indices_to_change)

            # --- 修改点：选择新值时，确保与原值不同 ---
            # c. 获取该列在上一行的原始值
            original_value = previous_settings[col_to_change_index]
            # d. 从所有可能的值中，排除掉原始值，得到允许的新值列表
            allowed_new_values = [v for v in all_options if v != original_value]
            # e. 从允许的新值列表中随机选择一个
            new_value = random.choice(allowed_new_values)
            # --- 结束修改点 ---

            # f. 基于上一行的设置，只改变选定的列
            current_settings = list(previous_settings) # 必须创建副本！
            current_settings[col_to_change_index] = new_value
            # g. 更新 'last_changed_col_index' 为这次改变的列
            last_changed_col_index = col_to_change_index

        # 将当前行的设置值添加到各自的列表中
        top_values.append(current_settings[0])
        central_values.append(current_settings[1])
        bottom_values.append(current_settings[2])

        # 更新 'previous_settings' 为当前行的设置，供下一行使用
        previous_settings = current_settings

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        'time': time_values,
        'top_setting': top_values,
        'central_setting': central_values,
        'bottom_setting': bottom_values
    })

    return df

# --- 使用新函数生成数据 ---



def generate_repeat():
    """生成一个符合描述的随机数据框"""
    num_rows = np.random.randint(10, 31)
    df = pd.DataFrame(columns=["time", "top_setting", "central_setting", "bottom_setting"])
    time = abs(np.random.normal(0.5,0.1))
    settings = [0, 0, 0]
    df.loc[0] = [time] + settings
    last_changed_col_index = np.random.randint(3)
    possible_values = [-2, -1, 1, 2]
    current_value = settings[last_changed_col_index]
    new_value = np.random.choice([v for v in possible_values if v != current_value])
    settings[last_changed_col_index] = new_value
    time += abs(np.random.normal(0.5,0.1))
    df.loc[1] = [time] + settings

    current_row_index = 2
    while current_row_index < num_rows:
        # 随机重复当前的设置
        repeat_count = np.random.randint(3,6)
        for _ in range(repeat_count):
            if current_row_index < num_rows:
                time += abs(np.random.normal(0.5,0.1))
                df.loc[current_row_index] = [time] + settings
                current_row_index += 1
            else:
                break

        if current_row_index < num_rows:
            # 随机选择一个与上次不同的列进行更改
            available_cols = [i for i in range(3) if i != last_changed_col_index]
            if not available_cols:
                # 如果所有列都已更改过，则随机选择一个
                changed_col_index = np.random.randint(3)
            else:
                changed_col_index = np.random.choice(available_cols)
            last_changed_col_index = changed_col_index

            # 确保新的值与之前的值不同
            current_value = settings[changed_col_index]
            possible_values = [-2, -1, 1, 2]
            new_value = np.random.choice([v for v in possible_values if v != current_value])
            settings[changed_col_index] = new_value

            time += abs(np.random.normal(5,3))
            df.loc[current_row_index] = [time] + settings
            current_row_index += 1

    df["top_setting"] = pd.Categorical(df["top_setting"], categories=[-2, -1, 0, 1, 2], ordered=True)
    df["central_setting"] = pd.Categorical(df["central_setting"], categories=[-2, -1, 0, 1, 2], ordered=True)
    df["bottom_setting"] = pd.Categorical(df["bottom_setting"], categories=[-2, -1, 0, 1, 2], ordered=True)
    return df

if __name__ == "__main__":
    list_of_reset = []
    list_of_onestep = []
    list_of_repeat = []
    for i in range(100):
        list_of_reset.append(generate_reset())
        list_of_onestep.append(generate_onestep())
        list_of_repeat.append(generate_repeat())
    print(list_of_reset[10])
    print(list_of_repeat[10])
    print(list_of_onestep[10])
