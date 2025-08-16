import pandas as pd

# 定义函数，将 decoded_domains 中的域名拆分并处理为 .eth 形式
def process_domains(row):
    # 将域名拆分为列表并加上 ".eth"
    domain_list = [domain.strip() + ".eth" for domain in row['decoded_domains'].split(',')]
    # 为每个域名生成一行，复制 address 和 data 列
    rows = [{'address': row['address'], 'data': row['data'], 'decoded_domains': domain} for domain in domain_list]
    return rows

# 处理Excel文件
def process_excel(file_path, output_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 确保存在 'address', 'data', 'decoded_domains' 列
    if 'address' not in df.columns or 'data' not in df.columns or 'decoded_domains' not in df.columns:
        raise ValueError("Excel 文件中缺少 'address', 'data' 或 'decoded_domains' 列")

    # 使用 apply 展开每行的域名，将每行域名拆分为多个新行
    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(process_domains(row))

    # 将生成的列表转换为新的 DataFrame
    new_df = pd.DataFrame(expanded_rows)

    # 保存处理后的 Excel 文件
    new_df.to_excel(output_path, index=False)

# 使用示例
if __name__ == "__main__":
    input_file = 'renew/3.解码.xlsx'  # 输入文件路径
    output_file = 'renew/4.解码后拆分.xlsx'  # 输出文件路径

    process_excel(input_file, output_file)
