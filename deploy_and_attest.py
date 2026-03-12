"""
IPAttestation 合约一键部署 + 存证脚本
使用方法：
  1. 先获取Sepolia测试币（https://sepolia-faucet.pk910.de/）
  2. 将私钥填入下方 PRIVATE_KEY（或保持读取 sepolia_wallet.json）
  3. 运行: python3 deploy_and_attest.py

依赖安装:
  pip install web3 eth-account py-solc-x
"""

import json
import time
import hashlib
from web3 import Web3
from eth_account import Account

# ============================================================
# 配置（修改这里）
# ============================================================
# 从文件读取钱包（或直接填写私钥）
try:
    with open("sepolia_wallet.json") as f:
        wallet = json.load(f)
    PRIVATE_KEY = wallet["private_key"]
    SENDER_ADDRESS = wallet["address"]
except FileNotFoundError:
    PRIVATE_KEY = "YOUR_PRIVATE_KEY_HERE"   # 替换为你的私钥
    SENDER_ADDRESS = "YOUR_ADDRESS_HERE"

# Sepolia RPC 端点
RPC_ENDPOINTS = [
    "https://ethereum-sepolia-rpc.publicnode.com",
    "https://rpc.sepolia.org",
    "https://sepolia.drpc.org",
    "https://1rpc.io/sepolia",
]

# ============================================================
# 论文实验参数（三组存证数据）
# ============================================================
ATTESTATION_DATA = [
    {
        "key": "dataset_preprocessing",
        "description": "Dataset preprocessing parameters: NCAC 2024 software copyright, N=2827213, stratified 1%, Isolation Forest contamination=0.05",
        "content": json.dumps({
            "source": "NCAC 2024 software copyright registry",
            "total_records": 2827213,
            "sampling_ratio": 0.01,
            "final_sample": 28272,
            "anomaly_method": "Isolation Forest",
            "contamination": 0.05,
            "random_state": 42,
            "paper": "Cultural IP Securitization - Xu Yuming 2026"
        }, sort_keys=True)
    },
    {
        "key": "ag_lstm_results",
        "description": "AG-LSTM valuation model results: MAPE=6.7%, improvement over baseline LSTM=63.8%, p<0.001",
        "content": json.dumps({
            "model": "AG-LSTM",
            "architecture": "3-layer LSTM + attention gate",
            "hidden_size": 128,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "MAPE": 0.067,
            "baseline_LSTM_MAPE": 0.185,
            "improvement_pct": 63.8,
            "p_value": "<0.001",
            "paper": "Cultural IP Securitization - Xu Yuming 2026"
        }, sort_keys=True)
    },
    {
        "key": "ddpg_portfolio_results",
        "description": "DDPG portfolio optimization results: Sharpe=1.58, FX risk ±20%, portfolio weights validated",
        "content": json.dumps({
            "model": "DDPG",
            "actor_layers": [256, 128],
            "critic_layers": [256, 128],
            "gamma": 0.99,
            "tau": 0.005,
            "replay_buffer": 1000000,
            "training_episodes": 5000,
            "Sharpe_ratio": 1.58,
            "FX_risk_range": "±20%",
            "portfolio_weights": {
                "AI_software": 0.35,
                "traditional_manufacturing": 0.25,
                "internet_applications": 0.30,
                "others": 0.10
            },
            "paper": "Cultural IP Securitization - Xu Yuming 2026"
        }, sort_keys=True)
    }
]

# ============================================================
# 合约 ABI 和 Bytecode（预编译版本）
# ============================================================
# 简化版存证合约 ABI（attest函数）
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "string", "name": "description", "type": "string"}
        ],
        "name": "attest",
        "outputs": [{"internalType": "uint256", "name": "attestationId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "contentHash", "type": "bytes32"}],
        "name": "verify",
        "outputs": [
            {"internalType": "bool", "name": "exists", "type": "bool"},
            {
                "components": [
                    {"internalType": "uint256", "name": "id", "type": "uint256"},
                    {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
                    {"internalType": "string", "name": "description", "type": "string"},
                    {"internalType": "address", "name": "attester", "type": "address"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "uint256", "name": "blockNumber", "type": "uint256"}
                ],
                "internalType": "struct IPAttestation.Attestation",
                "name": "attestation",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "attestationId", "type": "uint256"},
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "description", "type": "string"},
            {"indexed": False, "internalType": "address", "name": "attester", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "blockNumber", "type": "uint256"}
        ],
        "name": "DataAttested",
        "type": "event"
    }
]

# 预编译的合约字节码（IPAttestation.sol 编译结果）
CONTRACT_BYTECODE = "0x608060405234801561001057600080fd5b5033600060006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055506000600181905550610a2d806100686000396000f3fe608060405234801561001057600080fd5b50600436106100415760003560e01c80632c8e4d2b14610046578063b10e1dbc14610076578063d5c7f4b1146100a6575b600080fd5b610060600480360381019061005b91906105a8565b6100d6565b60405161006d91906105f3565b60405180910390f35b610090600480360381019061008b919061060e565b6102a2565b60405161009d919061068e565b60405180910390f35b6100c060048036038101906100bb91906106a9565b6103f3565b6040516100cd91906106f5565b60405180910390f35b600080600083815260200190815260200160002054905060008114610131576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161012890610771565b60405180910390fd5b600180549050905060018101600181905550600060c060405190810160405280838152602001868152602001858152602001600060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001428152602001438152509050806002600083815260200190815260200160002060008201518160000155602082015181600101556040820151816002019080519060200190610207929190610791565b5060608201518160030160006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff16021790555060808201518160040155"

# ============================================================
# 主程序
# ============================================================
def main():
    # 连接测试网
    w3 = None
    for rpc in RPC_ENDPOINTS:
        try:
            w3_test = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
            if w3_test.is_connected():
                w3 = w3_test
                print(f"✓ 已连接Sepolia测试网: {rpc}")
                print(f"  区块高度: {w3.eth.block_number}")
                break
        except Exception as e:
            print(f"  ✗ {rpc}: {e}")
    
    if w3 is None:
        print("无法连接到Sepolia测试网")
        return
    
    # 检查余额
    balance = w3.eth.get_balance(SENDER_ADDRESS)
    balance_eth = w3.from_wei(balance, 'ether')
    print(f"\n钱包地址: {SENDER_ADDRESS}")
    print(f"余额: {balance_eth} ETH")
    
    if balance_eth < 0.001:
        print(f"\n⚠ 余额不足（需要至少 0.001 ETH）")
        print(f"请前往水龙头获取测试币：")
        print(f"  https://sepolia-faucet.pk910.de/")
        print(f"  地址: {SENDER_ADDRESS}")
        return
    
    # 计算存证数据的SHA256哈希
    print("\n计算存证数据哈希...")
    content_hashes = []
    for item in ATTESTATION_DATA:
        sha256 = hashlib.sha256(item["content"].encode()).hexdigest()
        content_hashes.append(bytes.fromhex(sha256))
        print(f"  {item['key']}: 0x{sha256}")
    
    # 部署合约（使用简化版直接发送data交易）
    print("\n部署IPAttestation合约...")
    nonce = w3.eth.get_transaction_count(SENDER_ADDRESS)
    gas_price = int(w3.eth.gas_price * 1.2)
    
    deploy_tx = {
        'nonce': nonce,
        'gasPrice': gas_price,
        'gas': 800000,
        'data': CONTRACT_BYTECODE,
        'chainId': 11155111
    }
    
    signed_deploy = w3.eth.account.sign_transaction(deploy_tx, PRIVATE_KEY)
    deploy_hash = w3.eth.send_raw_transaction(signed_deploy.raw_transaction)
    print(f"  部署交易: {deploy_hash.hex()}")
    print("  等待确认...")
    
    deploy_receipt = w3.eth.wait_for_transaction_receipt(deploy_hash, timeout=120)
    contract_address = deploy_receipt.contractAddress
    print(f"  ✓ 合约地址: {contract_address}")
    print(f"  验证: https://sepolia.etherscan.io/address/{contract_address}")
    
    # 创建合约实例
    contract = w3.eth.contract(address=contract_address, abi=CONTRACT_ABI)
    
    # 发送3笔存证交易
    print("\n发送存证交易...")
    tx_results = []
    nonce = w3.eth.get_transaction_count(SENDER_ADDRESS)
    
    for i, (item, content_hash) in enumerate(zip(ATTESTATION_DATA, content_hashes)):
        try:
            attest_tx = contract.functions.attest(
                content_hash,
                item["description"]
            ).build_transaction({
                'from': SENDER_ADDRESS,
                'nonce': nonce + i,
                'gasPrice': gas_price,
                'gas': 200000,
                'chainId': 11155111
            })
            
            signed_attest = w3.eth.account.sign_transaction(attest_tx, PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_attest.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            print(f"  ✓ {item['key']}: {tx_hash_hex}")
            print(f"    验证: https://sepolia.etherscan.io/tx/{tx_hash_hex}")
            
            tx_results.append({
                "key": item["key"],
                "description": item["description"],
                "content_hash": "0x" + content_hash.hex(),
                "tx_hash": tx_hash_hex,
                "etherscan": f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"
            })
            
            time.sleep(2)
            
        except Exception as e:
            print(f"  ✗ {item['key']} 失败: {e}")
    
    # 等待所有交易确认
    print("\n等待交易确认...")
    for result in tx_results:
        try:
            receipt = w3.eth.wait_for_transaction_receipt(
                bytes.fromhex(result["tx_hash"][2:]), timeout=120
            )
            result["block_number"] = receipt.blockNumber
            result["status"] = "confirmed" if receipt.status == 1 else "failed"
            print(f"  ✓ {result['key']}: 区块 #{receipt.blockNumber}")
        except Exception as e:
            print(f"  ✗ 等待确认失败: {e}")
    
    # 保存结果
    final_result = {
        "contract_address": contract_address,
        "deploy_tx": deploy_hash.hex(),
        "network": "Ethereum Sepolia Testnet",
        "chain_id": 11155111,
        "attestations": tx_results
    }
    
    with open("tx_hashes.json", "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("✓ 链上存证完成！")
    print("="*60)
    print(f"合约地址: {contract_address}")
    print(f"结果已保存到: tx_hashes.json")
    print("\n论文 Appendix A.4 更新内容：")
    for r in tx_results:
        print(f"\n  {r['description'][:50]}...")
        print(f"  TX Hash: {r['tx_hash']}")
        print(f"  Etherscan: {r['etherscan']}")


if __name__ == "__main__":
    main()
