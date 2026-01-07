import pandas as pd
import pyreadstat

def split_strict_paired_events(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    严格地将成对的 "START_ITEM" 和 "END_ITEM" 及其之间的 "ACER_EVENT" 行分割为子 DataFrame。
    每个子 DataFrame 必须以 "START_ITEM" 开始，以 "END_ITEM" 结束，中间只包含零个或多个 "ACER_EVENT"。
    丢弃所有无法严格配对的行。

    Args:
        df: 包含 'event' 列的 Pandas DataFrame。

    Returns:
        包含所有严格配对的子 DataFrame 的列表。
    """
    sub_dfs = []
    start_indices = df[df['event'] == 'START_ITEM'].index.tolist()
    end_indices = df[df['event'] == 'END_ITEM'].index.tolist()

    start_idx = 0
    end_idx = 0

    while start_idx < len(start_indices) and end_idx < len(end_indices):
        if start_indices[start_idx] < end_indices[end_idx]:
            # 潜在的匹配
            temp_df = df.loc[start_indices[start_idx]:end_indices[end_idx]]
            # 检查在这个潜在的子 DataFrame 中是否包含其他的 START_ITEM (不包括第一个)
            if (temp_df['event'] == 'START_ITEM').iloc[1:].any():
                # 如果有夹在中间的 START_ITEM，则当前的 END_ITEM 可能不是我们想要的匹配
                # 我们移动到下一个 START_ITEM 寻找新的匹配
                start_idx += 1
            else:
                # 如果没有夹在中间的 START_ITEM，则这是一个有效的配对
                sub_dfs.append(temp_df.copy())
                start_idx += 1
                end_idx += 1
        elif start_indices[start_idx] > end_indices[end_idx]:
            # 当前的 END_ITEM 在 START_ITEM 之前，无法匹配，移动到下一个 END_ITEM
            end_idx += 1
        else:
            # start_indices[start_idx] == end_indices[end_idx]，理论上不应该发生
            start_idx += 1
            end_idx += 1

    return sub_dfs



def verify_sub_dfs_pattern(sub_dfs: list[pd.DataFrame]) -> pd.DataFrame | None:
    """
    检查子 DataFrame 列表中的每个元素是否满足以 "START_ITEM" 开始，以 "END_ITEM" 结束，
    中间只有零个或多个 "ACER_EVENT" 的规律。

    Args:
        sub_dfs: 包含 Pandas DataFrame 的列表。

    Returns:
        如果找到第一个不满足规律的子 DataFrame，则返回该子 DataFrame。
        如果所有子 DataFrame 都满足规律，则返回 None。
    """
    for i, sub_df in enumerate(sub_dfs):
        if sub_df.empty:
            print(f"警告：子 DataFrame {i+1} 为空。")
            continue

        first_event = sub_df['event'].iloc[0]
        last_event = sub_df['event'].iloc[-1]
        middle_events = sub_df['event'].iloc[1:-1]

        if first_event != 'START_ITEM':
            print(f"错误：子 DataFrame {i+1} 不以 'START_ITEM' 开始。")
            return sub_df
        if last_event != 'END_ITEM':
            print(f"错误：子 DataFrame {i+1} 不以 'END_ITEM' 结束。")
            return sub_df
        if not middle_events.isin(['ACER_EVENT']).all():
            print(f"错误：子 DataFrame {i+1} 中间包含非 'ACER_EVENT' 的事件。")
            return sub_df

    return None





def transform_sub_dfs(result: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    对子 DataFrame 列表中的每个元素进行以下转化：
    1. 找到最后一个 event_type 为 "apply" 或 "reset" 的行，将其 diag_state 修改为倒数第二行的 diag_state。
    2. 将 time 列的每一项都减去 START_ITEM 所在行的 time 值。
    3. 去除所有 event_type 不为 "apply" 或 "reset" 的行。
    4. 将修改后的元素存储到新的列表中。

    Args:
        result: 包含 Pandas DataFrame 的列表。

    Returns:
        包含转化后 Pandas DataFrame 的新列表。
    """
    transformed_dfs = []
    for df in result:
        df_copy = df.copy()  # 为了不修改原始 DataFrame

        # 1. 查找并修改 diag_state
        apply_reset_indices = df_copy[df_copy['event_type'].isin(['apply'])].index.tolist()
        apply_diag_indices = df_copy[df_copy['event_type'].isin(['Diagram'])].index.tolist()
        if apply_reset_indices and apply_diag_indices and len(df_copy) >= 2:
            last_apply_reset_index = apply_reset_indices[-1]
            last_apply_diag_index = apply_diag_indices[-1]
            second_last_diag_state = df_copy.loc[last_apply_diag_index]['diag_state']
            df_copy.loc[last_apply_reset_index, 'diag_state'] = second_last_diag_state

        # 2. 调整 time 列
        start_item_row = df_copy[df_copy['event'] == 'START_ITEM']
        if not start_item_row.empty:
            start_time = start_item_row['time'].iloc[0]
            df_copy['time'] = df_copy['time'] - start_time

        # 3. 筛选行
        filtered_df = df_copy[df_copy['event_type'].isin(['apply'])].copy()
        transformed_dfs.append(filtered_df)

    return transformed_dfs





#是否需要剔除重复的数据？

def remove_consecutive_duplicates(transformed_result: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    对子 DataFrame 列表中的每个元素进行修改：当相邻的多行 "top_setting", "central_setting",
    "bottom_setting" 三列均相同时，只保留第一行，删除后续相同的行。

    Args:
        transformed_result: 包含 Pandas DataFrame 的列表。

    Returns:
        包含修改后 Pandas DataFrame 的新列表。
    """
    modified_dfs = []
    for df in transformed_result:
        if df.empty:
            modified_dfs.append(df.copy())  # 处理空 DataFrame
            continue

        # 使用 shift() 方法比较相邻行
        condition = (df['top_setting'] == df['top_setting'].shift(-1)) & \
                    (df['central_setting'] == df['central_setting'].shift(-1)) & \
                    (df['bottom_setting'] == df['bottom_setting'].shift(-1))

        # 保留 condition 为 False 的行（即与上一行不相同的行）和第一行
        modified_df = df[~condition | df.index.isin([df.index[-1]])].copy()
        modified_dfs.append(modified_df)

    return modified_dfs






def create_time_series_with_result(result: list[pd.DataFrame]) -> list[list]:
    """
    将子 DataFrame 列表中的每个元素转换为一个包含时间序列和对应 diag_state 结果的列表。

    Args:
        result: 包含 Pandas DataFrame 的列表。

    Returns:
        一个最终列表，其中每个元素都是一个包含时间序列 DataFrame 和对应 diag_state 值的列表。
    """
    final_list = []
    for df in result:
        if not df.empty:
            # 1. 提取时间序列数据
            time_series_df = df[['time', 'top_setting', 'central_setting', 'bottom_setting']].copy()

            # 2. 获取结果 (diag_state)
            diag_state_result = df['diag_state'].iloc[-1] if 'diag_state' in df.columns else None

            # 3. 创建成对列表
            paired_list = [time_series_df, diag_state_result]
            final_list.append(paired_list)
        else:
            final_list.append([pd.DataFrame(columns=['time', 'top_setting', 'central_setting', 'bottom_setting']), None]) # 处理空 DataFrame

    return final_list



def transform_setting_data_handle_categorical_input(input_list):
    """
    将学生答题数据列表中的 dataframe 进行转换。
    转换内容：将 'top_setting', 'central_setting', 'bottom_setting' 三列
              (它们已经是 category 类型) 组合成一个新的分类变量。

    Args:
        input_list: 列表，每个元素是 [dataframe, label]。
                    dataframe 包含 'time', 'top_setting', 'central_setting',
                    'bottom_setting' 列。这些设置列的 dtype 是 'category'。

    Returns:
        list: 新的列表，每个元素是 [new_dataframe, label]。
              new_dataframe 包含 'time', 'combined_setting' 列。
              'combined_setting' 是由原始三列 category 编码而成的新分类变量 (0-124)，
              其数据类型为 'category'。
    """
    transformed_data_list = []

    for df ,label in input_list:



        # 步骤1: 从 category 列中获取其代表的数值 (例如，将 category 类型转换回 int)
        # 这样我们就可以对数值进行算术运算
        top_numeric = df['top_setting'].astype(int)
        central_numeric = df['central_setting'].astype(int)
        bottom_numeric = df['bottom_setting'].astype(int)

        # 步骤2: 对获取的数值进行映射和五进制编码
        # 将 -2,-1,0,1,2 映射到 0,1,2,3,4
        mapped_top = top_numeric + 2
        mapped_central = central_numeric + 2
        mapped_bottom = bottom_numeric + 2

        # 计算组合编码 (0-124)
        # 这个整数值唯一代表了 5*5*5=125 种可能的设置组合
        combined_setting_id = mapped_top * 25 + mapped_central * 5 + mapped_bottom

        # 构建新的 dataframe，只包含 time 和 新的组合分类变量
        new_df = pd.DataFrame({
            'time': df['time'],
            # combined_setting_id 列现在存储的是代表组合的整数 ID
            'combined_setting_id': combined_setting_id
        })

        # 步骤3: 将存储组合 ID 的列转换为 pandas 的 'category' 类型
        # 定义所有可能的组合 ID 作为新 category 列的类别，确保有 125 种
        all_possible_combined_categories = pd.RangeIndex(0, 125) # 创建从0到124的Index作为所有可能的类别
        new_df['combined_setting'] = pd.Categorical(
            new_df['combined_setting_id'],
            categories=all_possible_combined_categories,
            ordered=False # 组合后的类别通常不再有简单的线性顺序关系
        )

        # 可以选择删除临时的 combined_setting_id 列
        new_df = new_df[['time', 'combined_setting']]

        # 将新的 [dataframe, label] 对添加到结果列表
        transformed_data_list.append([new_df,label])

    return transformed_data_list
#还原编码
setting_categories = [-2, -1, 0, 1, 2]
# Assuming the original settings were ordered, we restore them as ordered categories
setting_dtype = pd.CategoricalDtype(categories=setting_categories, ordered=True)


def restore_from_transformed_element(transformed_element):
    """
    将一个转换后的数据元素 (来自 transformed_list) 还原为原始格式。

    Args:
        transformed_element: 列表，一个元素来自 transformed_list，格式为
                             [transformed_dataframe, label]。
                             transformed_dataframe 包含 'time', 'combined_setting' 列。
                             'combined_setting' 列是 dtype='category'，其类别是 0-124 的整数。

    Returns:
        list: 还原后的元素，格式为 [restored_dataframe, label]。
              restored_dataframe 包含 'time', 'top_setting', 'central_setting',
              'bottom_setting' 列。这三列是 dtype='category'，其类别是 -2到2。
    Raises:
        ValueError: 如果输入格式不正确。
    """
    if not isinstance(transformed_element, list) or len(transformed_element) != 2:
        raise ValueError("Input must be a list of two elements: [dataframe, label]")

    transformed_df, label = transformed_element

    if not isinstance(transformed_df, pd.DataFrame):
         raise ValueError("First element of the input list must be a pandas DataFrame")

    if not all(col in transformed_df.columns for col in ['time', 'combined_setting']):
         raise ValueError("Input DataFrame must contain 'time' and 'combined_setting' columns")

    # 确保 'combined_setting' 是 category 类型，并且其类别是整数 (0-124)
    if transformed_df['combined_setting'].dtype.name != 'category':
         raise ValueError("Input 'combined_setting' column must be of category dtype")

    # Access the underlying integer values (the categories themselves, which are 0-124)
    # Using .astype(int) is a robust way to get the integer representation of numeric categories
    combined_values = transformed_df['combined_setting'].astype(int)

    # 逆转五进制编码 (分解)
    # combined_setting_id = mapped_top * 25 + mapped_central * 5 + mapped_bottom * 1

    mapped_top = combined_values // 25         # 整数除法
    remainder = combined_values % 25           # 取余数
    mapped_central = remainder // 5
    mapped_bottom = remainder % 5              # 或者 remainder // 1, 但 % 5 更清晰表示末位

    # 逆转映射 (将 0-4 还原到 -2-2)
    top_setting_numeric = mapped_top - 2
    central_setting_numeric = mapped_central - 2
    bottom_setting_numeric = mapped_bottom - 2

    # 创建新的 DataFrame，包含原始 time 和还原后的三列设置
    restored_df = pd.DataFrame({
        'time': transformed_df['time'],
        'top_setting': top_setting_numeric,
        'central_setting': central_setting_numeric,
        'bottom_setting': bottom_setting_numeric
    })

    # 将还原后的设置列转换为 category 类型，使用之前定义的有序类别和 dtype
    for col in ['top_setting', 'central_setting', 'bottom_setting']:
        restored_df[col] = restored_df[col].astype(setting_dtype)


    # 返回还原后的 [dataframe, label] 元素
    return [restored_df, label]



if __name__ == "__main__":
    df, meta = pyreadstat.read_sav(r"E:\复旦大学\大四下\毕业论文\代码\rawdata\CBA_cp025q01_logs12_SPSS.sav")


    result = split_strict_paired_events(df)
    # 调用验证函数
    first_invalid_df = verify_sub_dfs_pattern(result)

    if first_invalid_df is not None:
        print("\n第一个不满足规律的子 DataFrame 是：")
        print(first_invalid_df)
    else:
        print("\n所有子 DataFrame 都符合规律。")

    # 调用转化函数
    transformed_result = transform_sub_dfs(result)

    final_result = remove_consecutive_duplicates(transformed_result)

    final_result_list = create_time_series_with_result(transformed_result)
