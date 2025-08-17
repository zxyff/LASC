import pandas as pd


def read_addresses(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=None, header=None)
    deposit_addresses = df['存'][0].str.lower().tolist()
    withdrawal_addresses = df['取'][0].str.lower().tolist()
    return deposit_addresses, withdrawal_addresses

def read_address_pairs(file_path):
    df = pd.read_excel(file_path, header=None)
    return df.values.tolist()

def match_addresses(deposit_addresses, withdrawal_addresses, address_pairs, output_file):
    results = []
    for pair in address_pairs:
        addr1, addr2 = pair[0].lower(), pair[1].lower()
        if addr1 in deposit_addresses and addr2 in withdrawal_addresses:
            results.append([pair[0], pair[1]])

    result_df = pd.DataFrame(results, columns=['存款地址', '取款地址'])
    result_df.to_excel(output_file, index=False)


def main():
    deposit_path = 'six_account_models.xlsx'
    address_pairs_path = 'address_pair/Definition_10.xlsx'
    output_path = 'address_pair/Definition_10_dewi.xlsx'

    # 读取存取地址
    deposit_addresses, withdrawal_addresses = read_addresses(deposit_path)
    # 读取地址对
    address_pairs = read_address_pairs(address_pairs_path)
    # 匹配并写入结果
    match_addresses(deposit_addresses, withdrawal_addresses, address_pairs, output_path)
    print("匹配完成，结果已写入", output_path)


if __name__ == "__main__":
    main()
