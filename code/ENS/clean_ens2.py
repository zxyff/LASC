import pandas as pd

file_path = 'renew/address.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')


result = []

for index, row in df.iterrows():
    address = str(row['address']).lower() if pd.notna(row['address']) else None
    ad1 = str(row['ad1']).lower() if pd.notna(row['ad1']) else None
    ad2 = str(row['ad2']).lower() if pd.notna(row['ad2']) else None

    if pd.isna(ad1) and pd.isna(ad2):
        continue

    if pd.isna(ad1) and ad2 != address:
        result.append([row['address'], row['ad2']])
    elif pd.isna(ad2) and ad1 != address:
        result.append([row['address'], row['ad1']])

    elif ad1 == ad2 and ad1 != address:
        result.append([row['address'], row['ad1']])

    elif address == ad1 and address != ad2:
        result.append([row['address'], row['ad2']])

    elif address == ad2 and address != ad1:
        result.append([row['address'], row['ad1']])

    elif address != ad1 and address != ad2 and ad1 != ad2:
        result.append([row['address'], row['ad1']])
        result.append([row['address'], row['ad2']])
        result.append([row['ad1'], row['ad2']])

result_df = pd.DataFrame(result, columns=['Address1', 'Address2'])
output_file = 'renew/address-renew-process.xlsx'
result_df.to_excel(output_file, index=False)


