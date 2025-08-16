import pandas as pd


# 读取存款和取款地址的 Excel 文件
def read_addresses(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=None, header=None)
    deposit_addresses = df['存'][0].str.lower().tolist()  # 获取存款地址列表并转换为小写
    withdrawal_addresses = df['取'][0].str.lower().tolist()  # 获取取款地址列表并转换为小写
    return deposit_addresses, withdrawal_addresses


# 读取地址对的 Excel 文件
def read_address_pairs(file_path):
    df = pd.read_excel(file_path, header=None)
    return df.values.tolist()  # 返回地址对列表


# 进行匹配并保存结果
def match_addresses(deposit_addresses, withdrawal_addresses, address_pairs, output_file):
    results = []
    for pair in address_pairs:
        addr1, addr2 = pair[0].lower(), pair[1].lower()  # 地址转换为小写
        if addr1 in deposit_addresses and addr2 in withdrawal_addresses:
            results.append([pair[0], pair[1]])  # 存款地址在前，取款地址在后，保持原格式

    # 创建 DataFrame 并导出到 Excel
    result_df = pd.DataFrame(results, columns=['存款地址', '取款地址'])
    result_df.to_excel(output_file, index=False)


def main():
    deposit_path = '六种模式存取款账户.xlsx'  # 第一份文件路径
    address_pairs_path = '完成的地址对/地址对-同域名不同续订人.xlsx'  # 第二份文件路径
    output_path = '完成的地址对/地址对-同域名不同续订人-分存取.xlsx'  # 输出文件路径

    # 读取存取地址
    deposit_addresses, withdrawal_addresses = read_addresses(deposit_path)
    # 读取地址对
    address_pairs = read_address_pairs(address_pairs_path)

    # 匹配并写入结果
    match_addresses(deposit_addresses, withdrawal_addresses, address_pairs, output_path)
    print("匹配完成，结果已写入", output_path)


if __name__ == "__main__":
    main()
