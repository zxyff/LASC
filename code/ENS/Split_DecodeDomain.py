import pandas as pd

def process_domains(row):

    domain_list = [domain.strip() + ".eth" for domain in row['decoded_domains'].split(',')]

    rows = [{'address': row['address'], 'data': row['data'], 'decoded_domains': domain} for domain in domain_list]
    return rows

def process_excel(file_path, output_path):

    df = pd.read_excel(file_path)

    if 'address' not in df.columns or 'data' not in df.columns or 'decoded_domains' not in df.columns:
        raise ValueError("Excel 文件中缺少 'address', 'data' 或 'decoded_domains' 列")

    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(process_domains(row))

    new_df = pd.DataFrame(expanded_rows)

    new_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    input_file = 'renew/3.decode.xlsx'
    output_file = 'renew/4.decode_split.xlsx'

    process_excel(input_file, output_file)
