#  Unveiling Ethereum Mixing Services Using Enhanced Graph Structure Learning

This is a Pytorch implementation of LASC, as described in the following:

> Unveiling Ethereum Mixing Services Using Enhanced Graph Structure Learning

## Requirements

**Hardware Configuration:**
All model training and testing were conducted on a machine with the following specifications:

- **CPU:** 20-core Intel Core i7-14700KF @ 3.4GHz
- **GPU:** Nvidia GeForce RTX 4070 Ti
- **RAM:** 32GB
- **Storage (HDD):** 2TB

**Software Environment:**

- Linux ubuntu 22.04
- Python 3.8
- PyTorch 1.8
- CUDA 11.1
- NumPy 1.24.4
- NetworkX 3.1



## Dataset



 We experimented with real transactions on the Ethereum mainnet from December 15, 2019, to August 8, 2022, during which Tornado Cash worked properly without sanctions. We collected raw transaction data using the Ethereum client Geth (https://geth.ethereum.org/) and Ethereum ETL (https://ethereum-etl.readthedocs.io/en/latest/). After the mixing data preparation phase (the first module of LASC), combined with the core contract account set, the ground truth dataset is derived from raw transaction data.
- Core Contract Account Set
- Ground Truth Dataset (949 mixing account pairs)

**Contract address set：**

| Contract                           | Address                                    |
| ---------------------------------- | ------------------------------------------ |
| 0.1 ETH                            | 0x12D66f87A04A9E220743712cE6d9bB1B5616B8Fc |
| 1 ETH                              | 0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936 |
| 10 ETH                             | 0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF |
| 100 ETH                            | 0xA160cdAB225685dA1d56aa342Ad8841c3b53f291 |
| Old Proxy                          | 0x905b63fff465b9ffbf41dea908ceb12478ec7601 |
| Proxy                              | 0x722122dF12D4e14e13Ac3b6895a86e84145b6967 |
| Rounter                            | 0xd90e2f925da726b50c4ed8d0fb90ad053324f31b |
| ENS: Base Registrar Implementation | 0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85 |
| ENS: Eth Name Service              | 0x314159265dD8dbb310642f98f50C066173C1259b |
| ENS: Bulk Renewal                  | 0xfF252725f6122A92551A5FA9a6b6bf10eb0Be035 |
| ENS: Registry with Fallback        | 0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e |
| ENS: Old ETH Registrar Controller  | 0x283Af0B28c62C092C9727F1Ee09c02CA627EB7F5 |





## How to Run

### 1. Mixing Data Preparation

#### RawData Collection:

Run [Geth](https://geth.ethereum.org/) to synchronize Ethereum blockchain data, and utilize [EthereumETL](https://ethereum-etl.readthedocs.io/) to parse the data. Extract the raw transactions for Tornado Cash and ENS respectively by matching with core contract account set.

**External Transaction Sample：**

| timestamp  | transactionHash                                              | from                                       | to                                         | value | gasLimit | gasPrice | gasUsed | callingFunction |
| ---------- | ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ----- | -------- | -------- | ------- | --------------- |
| 1576526361 | 0xcfa3a64a54e096eb23d808fa75f949999ec79eea2a90781c5ef31bc6686ef69a | 0x0039f22efb07a647557c7c5d17854cfd6d489ef3 | 0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc | 1E+17 | 1000000  | 1.38E+10 | 981887  | 0xb214faa5      |

**Internal Transaction Sample：**

| Transaction  Hash                                            | From                                       | TxTo                                       | Value |
| ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ----- |
| 0x9bb7303af6ce69085abc3d9f4f5b7884a90023fd6e5925cb6ffed9737ebff78c | 0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc | 0x0039f22efb07a647557c7c5d17854cfd6d489ef3 | 0     |



#### Transfer Path Restoration

Six usage patterns of mixing implementation on Tornado Cash are formalized to restore real transfer paths and mixing accounts. 

1. **Deposit Operations**:
   - **Pattern a** : Funds transfer directly from deposit accounts to mixing contracts. These transactions are identifiable easily from mixing contracts' external transactions.
   - **Pattern b**: Users deposit funds by invoking the router contract's deposit function. It requires reconstructing transfer paths by combining the external transactions of the router contract with the internal transactions of mixing contracts.
2. **Withdrawal Operations**:
   - **Pattern c**: Users withdraw funds directly from mixing contracts.
   - **Pattern d**: The relayer account set by the user initiates a withdrawal transaction, leading the mixing contract to pay the mixing fund and the reward to the withdrawal account and the relayer account, respectively.
   - **Pattern e**: Users directly interact with the old proxy/proxy/router contracts for withdrawal.
   - **Pattern f**: Relayer accounts on behalf of users interact with the old proxy/proxy/router contracts to make withdrawals.

**Data Processing Pipeline:**

1. **Initial Data Extraction**:
   Match the raw ledger transactions from the full node with the core contract accounts of Tornado Cash using `get0-transaction.py`.

2. **Pattern Matching**:

   - Patterns a & c extracted directly from mixing contracts' external transactions

   - Pattern d paths reconstructed by combining internal/external transactions
  
   - Pattern B restores the true transfer path by integrating the router's external transactions with the hybrid contract's internal transactions.

   - Patterns e, f processed through

     - interna_blockchain.py`: Link internal/external transactions by transaction hash

     - `withdraw-replacenew.py`: Update `from` and `to` fields in external transactions

     - `withdraw-special.py`: Filter transactions that meet specific Patterns.

3. **Final Output**:
   Generate a dataset of Tornado Cash real mixing transactions `6_pattern.xlsx`



#### Label Account Extraction AND Ground-Truth Dataset Construction

Given the massive size of the raw ENS dataset, a lightweight method via blockchain explorer is provided:

**1. ENS Controller-based Account Correlation Heuristic**

(1) **Data Acquisition:** Construct custom queries to call the Dune Analytics public database ([Dune Query](https://dune.com/queries)), retrieving a list of accounts for all ENS domain owners on the specified date(s).

```sql
SELECT DISTINCT owner
FROM (
    SELECT contract_address, cost, expires, label, bytearray_to_uint256(label) AS tokenid, owner, name, evt_block_number, evt_block_time, evt_index, evt_tx_hash
    FROM ethereumnameservice_ethereum.ETHRegistrarController_1_evt_NameRegistered
    WHERE evt_block_time BETWEEN TIMESTAMP '2019-12-01 00:00:00' AND TIMESTAMP '2022-08-31 23:59:59'
    UNION
    SELECT contract_address, cost, expires, label, bytearray_to_uint256(label) AS tokenid, owner, name, evt_block_number, evt_block_time, evt_index, evt_tx_hash
    FROM ethereumnameservice_ethereum.ETHRegistrarController_2_evt_NameRegistered
    WHERE evt_block_time BETWEEN TIMESTAMP '2019-12-01 00:00:00' AND TIMESTAMP '2022-08-31 23:59:59'
    UNION
    SELECT contract_address, cost, expires, label, bytearray_to_uint256(label) AS tokenid, owner, name, evt_block_number, evt_block_time, evt_index, evt_tx_hash
    FROM ethereumnameservice_ethereum.ETHRegistrarController_3_evt_NameRegistered
    WHERE evt_block_time BETWEEN TIMESTAMP '2019-12-01 00:00:00' AND TIMESTAMP '2022-08-31 23:59:59'
    UNION
    SELECT contract_address, baseCost + premium AS cost, expires, label, bytearray_to_uint256(label) AS tokenid, owner, name, evt_block_number, evt_block_time, evt_index, evt_tx_hash
    FROM ethereumnameservice_ethereum.ETHRegistrarController_4_evt_NameRegistered
    WHERE evt_block_time BETWEEN TIMESTAMP '2019-12-01 00:00:00' AND TIMESTAMP '2022-08-31 23:59:59'
) AS combined
```

(2) **Address Matching:** Match the retrieved ENS owner addresses against the set of real mixing accounts.

(3) **Intersection Handling:** Store the successfully matched accounts in the file `ens-common_data.xlsx`.

(4) **Domain Crawling:** For each accounts `{address}` in the intersection list, use the script `ENS_Controller-based.py` to access the ENS app page ([https://app.ens.domains/{address}](https://app.ens.domains/%7Baddress%7D)) and crawl the associated domain names under specific conditions.

**2. ENS Renewal-based Account Correlation Heuristic**

(1) **Transaction Data Acquisition:** Download ENS Bulk Renewal transactions occurring on the specified date(s) from etherscan.io.

(2) **Data Cleaning & Matching:** Clean the transaction data and match it against the real mixing account set.

(3) **Transaction Detail Query:** Using the transaction hash and block number of matched transactions, construct custom queries to call the Dune Analytics public database ([Dune Query](https://dune.com/queries)), retrieving the transaction input data (`data` field).

```sql
SELECT
  data
FROM
  ethereum.transactions
WHERE
  block_number IN ({{block_list}})
  AND hash IN ({{txhash_list}})
ORDER BY
  block_number ASC
```

(4) **Data Decoding:** Decode the retrieved `data` field using the script `decoding.py`.

(5) **Domain Processing:** Process the decoded domain names into the specified format using the script `Split_DecodeDomain.py`.

(6) **Domain Info Crawling:** Combine the **domains crawled from Method 1** and **domains processed in Step (5) of Method 2**, then feed them into the script `ENS_Renewal-based.py`. This script accesses the ENS app domain management page (format: [https://app.ens.domains/{domain}?tab=ownership](https://app.ens.domains/%7Bdomain%7D?tab=ownership)) to crawl manager and owner addresses for each domain.

(7) **Data Cleaning & Deduplication:** Use the script `clean_ens2.py` to **clean and deduplicate all crawled manager/owner addresses from the combined dataset**.

(8) **Final Matching:** Use the script `matching_difDEorWI.py` to **match the consolidated cleaned address set** against the genuine mixed user set, yielding the final ground truth dataset.



### 2.Mixing Transfer Graph Construction

#### Graph Structure

Mixing Data Graph is a directed graph
$$
\mathcal{G}_{D}=(\mathcal{V}_{C},\mathcal{V}_{U},\mathcal{V}_{N},\mathcal{E}_{CU},\mathcal{E}_{UU},\mathcal{E}_{UN},\mathcal{E}_{DW})
$$
Mixing Transfer Graph is a directed graph
$$
\mathcal{G}_{T}=(\mathcal{V}_{U},\mathcal{V}_{N},\mathcal{E}_{UU},\mathcal{E}_{UN})
$$

#### Features and dimensionality reduction

The original node features with 100 dimensions, including four categories: pattern, quantity, time, and amount. After standardizing these features, we identify and delete redundant items based on the Pearson correlation coefficient and variance inflation factor (VIF). Principal component analysis (PCA) is applied to the remaining 75-dimensional features, which finally outputs 32-dimensional features to represent the most informative aspects of the original features.

![image-20250817160735876](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250817160735876.png)

### 3. Mixing Accounts Correlation

code/LASC

```python
python main.py
```






