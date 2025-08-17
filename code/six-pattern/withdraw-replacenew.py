import pandas as pd
# 加载数据集
file_path = 'dataset/processing/Proxy-router/router-withdraw.csv'
data = pd.read_csv(file_path, low_memory=False)
# 提取相关的外部和内部交易列
external_cols = data.columns[:18] # 外部交易列
internal_cols = data.columns[18:] # 内部交易列
# 将数据集分为外部和内部交易
external_tx = data[external_cols]
internal_tx = data[internal_cols]
grouped_internal_tx = internal_tx.groupby('Transaction Hash')

single_internal_tx_list = []
multiple_internal_tx_list = []
# 根据内部交易的数量处理替换逻辑的函数
def process_unique_transactions(external_tx_row):
    tx_hash = external_tx_row['transactionHash']
    related_internal_tx = grouped_internal_tx.get_group(tx_hash)
    if len(related_internal_tx) == 1:
        # 单个内部交易，直接替换
        new_row = external_tx_row.copy()
        new_row['from'] = related_internal_tx['From'].values[0]
        new_row['to'] = related_internal_tx['TxTo'].values[0]
        single_internal_tx_list.append(new_row)
    elif len(related_internal_tx) > 1: # 多个内部交易的情况
        # 找出 `Value_OUT(ETH)` 最大的交易并替换
        new_row = external_tx_row.copy()
        larger_tx = related_internal_tx.loc[related_internal_tx['Value_OUT(ETH)'].idxmax()]
        new_row['from'] = larger_tx['From']
        new_row['to'] = larger_tx['TxTo']
        multiple_internal_tx_list.append(new_row)

external_tx.apply(process_unique_transactions, axis=1)
# 将单个和多个内部交易的外部交易数据分别保存到新的 CSV 文件
single_tx_file_path = 'dataset/processing/Proxy-router/e-processed_qu_router.csv'
multiple_tx_file_path = 'dataset/processing/Proxy-router/f-processed_qu_router.csv'
single_internal_tx_df = pd.DataFrame(single_internal_tx_list)
multiple_internal_tx_df = pd.DataFrame(multiple_internal_tx_list)
single_internal_tx_df.to_csv(single_tx_file_path, index=False)
multiple_internal_tx_df.to_csv(multiple_tx_file_path, index=False)
