# CIPS-Framework

## A Robust AI-Blockchain Integrated Framework for Cultural Intellectual Property Securitization

> From Multi-modal Authentication to Dynamic Portfolio Optimization

### Overview

This repository contains the source code, smart contracts, and on-chain attestation data accompanying the research paper:

**"A Robust AI-Blockchain Integrated Framework for Cultural Intellectual Property Securitization: From Multi-modal Authentication to Dynamic Portfolio Optimization"**

*Yuming Xu, Wei Jiang*, Zilin Qin*

The framework proposes an end-to-end AI-Blockchain integrated solution for Cultural IP Securitization (CIPS), addressing three core bottlenecks: ambiguous ownership, valuation biases, and insufficient risk management.

### Repository Structure

```
CIPS-Framework/
├── README.md                    # This file
├── financial_simulator.py       # AG-LSTM valuation & DDPG portfolio optimization simulator
├── IPAttestation.sol            # Solidity smart contract for on-chain IP attestation
├── deploy_and_attest.py         # Deployment & attestation script for Ethereum Sepolia testnet
├── tx_hashes.json               # On-chain transaction hashes for data attestation
└── OnChain_Complete_Guide.md    # Complete guide for on-chain verification
```

### Key Components

#### 1. Financial Simulator (`financial_simulator.py`)
- **AG-LSTM Valuation Model**: Attention-Gated LSTM with Isolation Forest preprocessing for IP asset valuation
- **DDPG Portfolio Optimization**: Deep Deterministic Policy Gradient for dynamic portfolio management
- **Monte Carlo Simulation**: Stress testing under exchange rate fluctuations

#### 2. Smart Contract (`IPAttestation.sol`)
- Solidity-based IP attestation contract for Ethereum
- On-chain timestamping of experimental parameter hashes (SHA-256)
- Event logging for transparent and verifiable data provenance

#### 3. Deployment & Attestation (`deploy_and_attest.py`)
- Automated deployment to Ethereum Sepolia testnet
- Batch attestation of experimental parameters
- Transaction hash recording and verification

### On-Chain Verification

The experimental parameters of this study have been timestamped on the **Ethereum Sepolia testnet**. The attestation details can be verified using the transaction hashes provided in `tx_hashes.json`.

For step-by-step verification instructions, please refer to `OnChain_Complete_Guide.md`.

### Requirements

```
Python >= 3.8
numpy
pandas
matplotlib
web3
solcx
```

### Installation

```bash
pip install numpy pandas matplotlib web3 py-solc-x
```

### Usage

```bash
# Run the financial simulator
python financial_simulator.py

# Deploy smart contract and perform attestation (requires Sepolia testnet ETH)
python deploy_and_attest.py
```

### Citation

If you find this work useful, please cite:

```bibtex
@article{xu2025cips,
  title={A Robust AI-Blockchain Integrated Framework for Cultural Intellectual Property Securitization: From Multi-modal Authentication to Dynamic Portfolio Optimization},
  author={Xu, Yuming and Jiang, Wei and Qin, Zilin},
  year={2025}
}
```

### License

This project is released for academic research purposes.

### Contact

- **Corresponding Author**: Wei Jiang (jiangwei@qdu.edu.cn)
- School of Computer Science and Technology, Qingdao University, Qingdao 266071, China
