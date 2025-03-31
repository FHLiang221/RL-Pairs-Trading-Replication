# Reinforcement Learning Pair Trading: A Dynamic Scaling Approach

This project replicates the paper ["Reinforcement Learning Pair Trading: A Dynamic Scaling Approach"](https://doi.org/10.3390/jrfm17120555) by Hongshen Yang and Avinash Malik (2024).

## Project Overview

This research investigates whether Reinforcement Learning (RL) can enhance decision-making in cryptocurrency algorithmic trading compared to traditional methods. The authors combine RL with pair trading, a statistical arbitrage technique that exploits price differences between statistically correlated assets.

### Key Approaches

- **Traditional Pair Trading**: Uses statistical thresholds to determine trading signals
- **RL1**: Reinforcement learning for trade timing decisions
- **RL2**: Reinforcement learning for both timing and investment quantity decisions

## Paper Summary

### Motivation
- Cryptocurrency markets are extremely volatile with ~$70 billion traded daily
- Traditional rule-based pair trading lacks flexibility to adapt to volatile markets
- Reinforcement learning could potentially optimize both trade timing and investment quantities

### Methodology
The paper introduces two RL approaches:
1. **RL1**: RL agent decides only on trade timing (when to trade)
2. **RL2**: RL agent decides both trade timing and investment quantity (when and how much to trade)

They designed:
- Custom RL environments for pair trading
- Novel reward shaping mechanisms 
- Specialized observation and action spaces

### Key Results
- Traditional pair trading: 8.33% annualized profit
- RL1 (A2C algorithm): 9.94% annualized profit
- RL2 (A2C algorithm): 31.53% annualized profit

## Repository Structure

- `/data`: Raw and processed cryptocurrency data
- `/notebooks`: Jupyter notebooks for analysis
- `/src`: Source code for data collection and processing
- `/docs`: Presentation materials and documentation

## Data Collection

This project uses BTC-EUR and BTC-GBP trading data at 1-minute intervals from Binance. The data spans from October 2023 to December 2023, with:
- Formation period: October-November 2023
- Testing period: December 2023

## Project Status

Current progress:
- [x] Understanding paper methodology
- [x] Setting up repository structure
- [x] Data collection and processing
- [x] Basic exploratory data analysis
- [ ] Implementation of traditional pair trading
- [ ] RL environment creation
- [ ] RL agent implementation and training
- [ ] Comparative evaluation

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, statsmodels

### Installation
```bash
git clone https://github.com/FHLiang221/RL-Pairs-Trading-Replication.git
cd your-repository-name
pip install -r requirements.txt