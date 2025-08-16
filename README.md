#  Unveiling Ethereum Mixing Services Using Enhanced Graph Structure Learning



## Requirements

**硬件配置:**
我们在一台配备以下硬件的 Linux 系统机器上进行了所有模型的训练和测试：

- **CPU:** 20 核 Intel Core i7-14700KF @ 3.4GHz
- **GPU:** Nvidia GeForce RTX 4070 Ti
- **内存 (RAM):** 32GB
- **存储 (HDD):** 2TB

**软件环境:**

- **操作系统:** Linux
- **编程语言:** Python 3.8
- **深度学习框架:** PyTorch 1.9
- **GPU 计算平台:** CUDA 11.1
- **科学计算库:** NumPy 1.24.4
- NetworkX 3.1

*(注: 此列表仅包含部分依赖项)*



**Hardware Configuration:**
All model training and testing were conducted on a Linux machine with the following specifications:

- **CPU:** 20-core Intel Core i7-14700KF @ 3.4GHz
- **GPU:** Nvidia GeForce RTX 4070 Ti
- **RAM:** 32GB
- **Storage (HDD):** 2TB

**Software Environment:**

- **Operating System:** Linux
- **Programming Language:** Python 3.8
- **Deep Learning Framework:** PyTorch 1.9
- **GPU Computing Platform:** CUDA 11.1
- **Scientific Computing Library:** NumPy 1.24.4
- NetworkX 3.1

*(Note: This list represents a portion of the dependencies)*

## Dataset

我们从 2019 年 12 月 15 日至 2022 年 8 月 8 日在以太坊主网上对真实交易进行了实验。在此期间，Tornado Cash 运行正常，没有受到任何制裁。我们从以太坊客户端 Geth 收集了约 817,500 笔原始 Tornado Cash 交易，其中内部交易 517,733 笔，外部交易 299,701 笔，数据收集过程遵循了第六节中提出的方法。表四展示了 Tornado Cash 核心合约的收集结果。如表四所示，43,218 个账户发起了 153,073 笔存款交易，而只有 4151 个账户发起了 138,278 笔取款交易。在混合数据准备阶段之后，我们得到了一个包含 272,236 条记录和 83,089 个账户的真实混合交易数据集，以及一个包含 **949 个混合账户对的真实数据集**。

 We experimented with real transactions on the Ethereum mainnet from December 15, 2019, to August 8, 2022, during which Tornado Cash worked properly without sanctions. Approximately 817,500 raw Tornado Cash transactions were collected from the Ethereum client Geth, with internal and external transactions of 517,733 and 299,701, respectively, following the components of the proposed method in Section VI. Table IV shows the collection results of Tornado Cash core contracts. As seen in Table IV, 43,218 accounts initiated 153,073 deposit transactions, while only 4151 accounts created 138,278 withdrawal transactions. After the mixing data preparation stage, we derived a real mixing transaction dataset containing 272,236 records and 83,089 accounts, as well as **a**
**ground-truth dataset of 949 mixing account pairs**.

合约地址集：

| 合约                               | 地址                                       |
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

1. Mixing Data Preparation

   RawData Collection:运行 Geth(Geth 链接)同步区块数据，并利用 EthereumETL(ETL链接)得到解析数据;利用合约地址集进行匹配，分别得到 TC和 ENS 原始数据集。
   外部交易格式内部交易格式:
   -Transfer Path Restoration:6 个算法-》真实用户
   Label Account Extraction+ Ground-Truth Dataset Construction由于 ENS 原始数据集庞大，提供借助区块链浏览器数据的标签提取简易方法

   

2. Mixing Transfer Graph Construction手动，图结构，原始 100 维特征，再用Pearson 系数+VIF到75 维，PCA降到 32维

3.  Mixing Accounts Correlation

   ```python
   python main.py
   ```

   服务器，+拼接

   


   
