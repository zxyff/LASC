#  Unveiling Ethereum Mixing Services Using Enhanced Graph Structure Learning

This is a Pytorch implementation of LASC, as described in the following:

> Unveiling Ethereum Mixing Services Using Enhanced Graph Structure Learning

## Requirements

**Hardware Configuration:**
All model training and testing were conducted on a Linux machine with the following specifications:

- **CPU:** 20-core Intel Core i7-14700KF @ 3.4GHz
- **GPU:** Nvidia GeForce RTX 4070 Ti
- **RAM:** 32GB
- **Storage (HDD):** 2TB

**Software Environment:**

- **Operating System:** Linux
- Python 3.8
- PyTorch 1.8
- CUDA 11.1
- NumPy 1.24.4
- NetworkX 3.1

*(Note: This list represents a portion of the dependencies)*

## Dataset

We experimented with real transactions on the Ethereum mainnet from December 15, 2019, to August 8, 2022, during which Tornado Cash worked properly without sanctions. Approximately 817,500 raw Tornado Cash transactions were collected from the Ethereum client Geth, with internal and external transactions of 517,733 and 299,701, respectively, following the components of the proposed method in Section VI. Table IV shows the collection results of Tornado Cash core contracts. As seen in Table IV, 43,218 accounts initiated 153,073 deposit transactions, while only 4151 accounts created 138,278 withdrawal transactions. After the mixing data preparation stage, we derived a real mixing transaction dataset containing **272,236 records** and **83,089 accounts**, as well as **a**
**ground-truth dataset of 949 mixing account pairs**.

Contract address set：

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

Run [Geth](https://geth.ethereum.org/) to synchronize Ethereum blockchain data, and utilize [EthereumETL](https://ethereum-etl.readthedocs.io/) to parse the data. By matching against contract address sets, extract the raw datasets for Tornado Cash and ENS respectively. Sample raw data is shown below。

**external transaction example：**

| lockNumber | timestamp | transactionHash                                              | from                                       | to                                         | toCreate | fromIsContract | toIsContract | value | gasLimit | gasPrice | gasUsed | callingFunction | isError | eip2718type | baseFeePerGas | maxFeePerGas | maxPriorityFeePerGas |
| ---------- | --------- | ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------ | -------- | -------------- | ------------ | ----- | -------- | -------- | ------- | --------------- | ------- | ----------- | ------------- | ------------ | -------------------- |
| 9117152    | 1.58E+09  | 0xcfa3a64a54e096eb23d808fa75f949999ec79eea2a90781c5ef31bc6686ef69a | 0x0039f22efb07a647557c7c5d17854cfd6d489ef3 | 0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc | None     | 0              | 1            | 1E+17 | 1000000  | 1.38E+10 | 981887  | 0xb214faa5      | None    | None        | None          | None         | None                 |

**internal  transaction example：**

| Transaction Hash                                             | Blockno | UnixTimestamp | DateTime (UTC) | ParentTxFrom                               | ParentTxTo                                 | ParentTxETH_Value | From                                       | TxTo                                       | ContractAddress | Value_IN(ETH) | Value_OUT(ETH) | CurrentValue @ $3114.81135579845/Eth | Historical $Price/Eth | Status | ErrCode | Type |
| ------------------------------------------------------------ | ------- | ------------- | -------------- | ------------------------------------------ | ------------------------------------------ | ----------------- | ------------------------------------------ | ------------------------------------------ | --------------- | ------------- | -------------- | ------------------------------------ | --------------------- | ------ | ------- | ---- |
| 0x9bb7303af6ce69085abc3d9f4f5b7884a90023fd6e5925cb6ffed9737ebff78c | 9117176 | 1.58E+09      | ######         | 0x0039f22efb07a647557c7c5d17854cfd6d489ef3 | 0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc | 0                 | 0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc | 0x0039f22efb07a647557c7c5d17854cfd6d489ef3 | 0               | 0.1           | 311.4811       | 132.68                               | 0                     |        | call    |      |

#### Transfer Path Restoration

We innovatively formalize Tornado Cash's fund mixing operations into six usage patterns:

1. **Deposit Operations**:
   - **Pattern a** : Funds transfer directly from deposit accounts to mixing contracts. These transactions are identifiable directly from mixing contracts' external transactions.
   - **Pattern b **: Users deposit funds by invoking router contracts' deposit functions, requiring reconstruction of full transfer paths by combining router contracts' external transactions with mixing contracts' internal transactions.
2. **Withdrawal Operations**:
   - **Pattern c**: Users withdraw funds directly from mixing contracts.
   - **Pattern d**: Users transfer funds to relayer accounts, which then initiate withdrawal transactions. Mixing contracts distribute mixed funds to withdrawal accounts and fees to relayers.
   - **Pattern e**: Withdrawals processed through router contracts.
   - **Pattern f **: Collaborative withdrawal mechanism combining relayers and routers.

**Data Processing Pipeline:**

1. **Initial Data Extraction**:
   Match real ledger transactions (including users, agents, routers) based on four mixing pool addresses using `get0-transaction.py`.

2. **Pattern Matching**:

   - Patterns a & c extracted directly from mixing contracts' external transactions

   - Pattern d paths reconstructed by combining internal/external transactions

   - Patterns b, e, f processed through code logic

     - interna_blockchain.py`: Link internal/external transactions by hash

     - `withdraw-replacenew.py`: Update `from` and `to` fields in external transactions

     - `withdraw-special.py`: Filter transactions meeting specific criteria

3. **Final Output**:
   Generated reconstructed transaction path dataset `6_pattern.xlsx`



#### Label Account Extraction AND Ground-Truth Dataset Construction

Given the massive size of the raw ENS dataset, a lightweight label extraction method leveraging blockchain explorer data is provided:

**（1）ENS Controller-based Account Correlation Heuristic**

1. **Data Acquisition:** Construct custom queries to call the Dune Analytics public database ([Dune Query: 4060770](https://dune.com/queries/4060770)), retrieving a list of addresses for all ENS domain owners on the specified date(s).
2. **Address Matching:** Match the retrieved ENS owner addresses against the set of genuine mixed accounts.
3. **Intersection Handling:** Store the successfully matched intersection addresses in the file `ens-common_data.xlsx`.
4. **Domain Crawling:** For each address `{address}` in the intersection list, use the script `ENS_Controller-based.py` to access the ENS app page ([https://app.ens.domains/{address}](https://app.ens.domains/%7Baddress%7D)) and crawl the associated domain names under specific conditions.

**（2）ENS Renewal-based Account Correlation Heuristic**

1. **Transaction Data Acquisition:** Download ENS Bulk Renewal transactions occurring on the specified date(s) from Etherscan.
2. **Data Cleaning & Matching:** Clean the transaction data and match it against the genuine transaction account set.
3. **Transaction Detail Query:** Using the transaction hash and block number of matched transactions, construct custom queries to call the Dune Analytics public database ([Dune Query: 4095696](https://dune.com/queries/4095696)), retrieving the transaction input data (`data` field).
4. **Data Decoding:** Decode the retrieved `data` field using the script `decoding.py`.
5. **Domain Processing:** Process the decoded domain names into the specified format using the script `Split_DecodeDomain.py`.
6.  **Domain Info Crawling:** Combine **domains crawled from Method (1)** and **domains processed in Step (5) of Method (2)**, then feed them into the script `ENS_Renewal-based.py`. This script accesses the ENS app domain management page (format: [https://app.ens.domains/{domain}?tab=ownership](https://app.ens.domains/%7Bdomain%7D?tab=ownership)) to crawl manager and owner addresses for each domain.
7. **Data Cleaning & Deduplication:** Use the script `clean_ens2.py` to **clean and deduplicate all crawled manager/owner addresses from the combined dataset**.
8. **Final Matching:** Use the script `matching_difDEorWI.py` to **match the consolidated cleaned address set** against the genuine mixed user set, yielding the final ground truth dataset.


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

The original node features with 100 dimensions, including four categories: pattern, quantity, time, and amount. After standardizing these features, we identify and delete redundant items based on the Pearson correlation coefficient and variance inflation factor (VIF). Principal component analysis (PCA) is applied to the remaining 75-dimensional features, which finally outputs 32-dimensional features to represent the most informative aspects
of the original features.

### 3. Mixing Accounts Correlation

code/LASC

```python
python main.py
```


