# import pandas as pd

# # 读取 Parquet 文件
# df = pd.read_parquet('/data2/yifan/datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet', engine='pyarrow')

# # 删除 'text' 列
# df = df.drop(columns=['text'])

# # 将 'input' 列设置为 'input' + 'instruction'
# df['instruction'] = df['instruction'].fillna("")
# df['input'] = df['instruction'] + " " + df['input']

# # 删除 'instruction' 列
# df = df.drop(columns=['instruction'])

# # 保存修改后的 DataFrame 到新的 Parquet 文件
# df.to_parquet('/data2/yifan/datasets/reproduced_alpaca/data.parquet', engine='pyarrow')


import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/data2/yifan/datasets/reproduced_alpaca/data.parquet', engine='pyarrow')

# 打印每一列的名字
print("列名列表：")
for col in df.columns:
    print(col)

# 打印某一列的前五行数据
column_name = 'input'  # 替换为你想查看的列名
print(df[column_name][6])
