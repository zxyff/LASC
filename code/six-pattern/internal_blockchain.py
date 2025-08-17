import pandas as pd

# 第一步：加载两个数据文件
file_path = '/path/to/first/excel/file.xlsx'
df_successful_transactions_pre = pd.read_excel(file_path, sheet_name='成功交易-存+取-前')

# 加载第二个CSV文件
csv_file_path = '/path/to/second/csv/file.csv'
df_internal_transactions = pd.read_csv(csv_file_path)

# 第二步：合并数据
merged_transactions = pd.merge(df_successful_transactions_pre, df_internal_transactions,
                               left_on='Txhash', right_on='Transaction Hash', how='inner')

# 第三步：保存合并后的数据
output_file_path = '/path/to/output/merged_transactions.csv'
merged_transactions.to_csv(output_file_path, index=False)
