import pandas as pd
file_path = 'dataset/processing/Proxy-router/router-withdraw.csv'
data = pd.read_csv(file_path, low_memory=False)

external_cols = data.columns[:18] # 外部交易列
internal_cols = data.columns[18:] # 内部交易列

external_tx = data[external_cols]
internal_tx = data[internal_cols]

grouped_internal_tx = internal_tx.groupby('Transaction Hash')

selected_transactions = pd.DataFrame(columns=['transactionHash', 'from', 'internalFrom', 'internalTo'])
# 遍历外部交易数据
for index, external_row in external_tx.iterrows():
    tx_hash = external_row['transactionHash']
    external_from = external_row['from']
    # 获取相关的内部交易
    if tx_hash in grouped_internal_tx.groups:
        related_internal_tx = grouped_internal_tx.get_group(tx_hash)
        # 只处理对应两个内部交易的外部交易
        if len(related_internal_tx) == 2:
            internal_tx_to_list = related_internal_tx['TxTo'].tolist()
            if external_from not in internal_tx_to_list:
                values_out = related_internal_tx['Value_OUT(ETH)'].tolist()
                if values_out[0] != values_out[1]:
                    # 选择 Value_OUT(ETH) 更大的交易
                    larger_index = values_out.index(max(values_out))
                    larger_internal_tx = related_internal_tx.iloc[larger_index]
                    selected_transactions = pd.concat([selected_transactions, pd.DataFrame({
                        'transactionHash': [tx_hash],
                        'from': [external_from],
                        'internalFrom': [larger_internal_tx['From']],
                        'internalTo': [larger_internal_tx['TxTo']]
                    })], ignore_index=True)
# 去重
unique_selected_transactions = selected_transactions.drop_duplicates(subset=['transactionHash'])
final_selected_transactions_file_path = 'dataset/processing/Proxy-router/processed_special_router.csv'
unique_selected_transactions.to_csv(final_selected_transactions_file_path, index=False)
print("数据处理完成。")