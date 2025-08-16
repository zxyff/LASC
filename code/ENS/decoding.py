import pandas as pd
from eth_abi.abi import decode  # 新的导入方式
from eth_utils import decode_hex


# 定义ABI类型，用于解码域名数组
def decode_domain_names(encoded_data):
    try:
        # 移除函数选择器，保留编码的数据部分
        data = encoded_data[10:]

        # 解析参数 types 定义：string[]（域名数组）、uint256（续租时长）
        decoded_params = decode(['string[]', 'uint256'], decode_hex(data))

        # 返回域名数组（第一个参数）
        return decoded_params[0]
    except Exception as e:
        return str(e)


# 读取Excel文件
def process_excel(file_path, output_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 确保存在 'data' 列
    if 'data' not in df.columns:
        raise ValueError("Excel 文件中缺少 'data' 列")

    # 新增一列 'decoded_domains' 用于存储解码后的域名
    df['decoded_domains'] = df['data'].apply(lambda x: ', '.join(decode_domain_names(x)))

    # 保存处理后的 Excel 文件
    df.to_excel(output_path, index=False)


# 使用示例
if __name__ == "__main__":
    input_file = 'renew/data完整.xlsx'  # 输入文件路径
    output_file = 'renew/解码.xlsx'  # 输出文件路径

    process_excel(input_file, output_file)
