import pandas as pd

# 读取Excel文件
file_path = 'renew/地址对-爬.xlsx'  # 将your_excel_file.xlsx替换为你的文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1')  # 替换成你的子表名称

# 存储结果的列表
result = []

# 遍历每一行数据
for index, row in df.iterrows():
    address = str(row['address']).lower() if pd.notna(row['address']) else None
    ad1 = str(row['ad1']).lower() if pd.notna(row['ad1']) else None
    ad2 = str(row['ad2']).lower() if pd.notna(row['ad2']) else None

    # 如果ad1和ad2均为空，舍弃
    if pd.isna(ad1) and pd.isna(ad2):
        continue

    # 如果ad1和ad2有一个为空，且非空的与address不同
    if pd.isna(ad1) and ad2 != address:
        result.append([row['address'], row['ad2']])
    elif pd.isna(ad2) and ad1 != address:
        result.append([row['address'], row['ad1']])

    # 如果ad1和ad2均非空且相同，且与address不同
    elif ad1 == ad2 and ad1 != address:
        result.append([row['address'], row['ad1']])

    # 如果address和ad1相同且和ad2不同
    elif address == ad1 and address != ad2:
        result.append([row['address'], row['ad2']])

    # 如果address和ad2相同且和ad1不同
    elif address == ad2 and address != ad1:
        result.append([row['address'], row['ad1']])

    # 如果address和ad1和ad2三者均不同
    elif address != ad1 and address != ad2 and ad1 != ad2:
        result.append([row['address'], row['ad1']])
        result.append([row['address'], row['ad2']])
        result.append([row['ad1'], row['ad2']])

# 将结果保存到新的Excel文件中
result_df = pd.DataFrame(result, columns=['Address1', 'Address2'])
output_file = 'renew/地址对-renew-处理后.xlsx'
result_df.to_excel(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")

