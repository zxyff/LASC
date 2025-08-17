import pandas as pd
from eth_abi.abi import decode
from eth_utils import decode_hex

# 定义ABI类型，用于解码域名数组
def decode_domain_names(encoded_data):
    try:
        data = encoded_data[10:]

        decoded_params = decode(['string[]', 'uint256'], decode_hex(data))

        return decoded_params[0]
    except Exception as e:
        return str(e)

def process_excel(file_path, output_path):
    df = pd.read_excel(file_path)

    if 'data' not in df.columns:
        raise ValueError("Excel 文件中缺少 'data' 列")

    df['decoded_domains'] = df['data'].apply(lambda x: ', '.join(decode_domain_names(x)))

    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    input_file = 'renew/data.xlsx'
    output_file = 'renew/decode.xlsx'

    process_excel(input_file, output_file)
