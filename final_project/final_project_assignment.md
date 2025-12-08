# Final Project: Deep Learning Enhanced Systematic Equity Strategy

## Overview

In this project, you will enhance an existing systematic equity strategy using deep learning techniques. The baseline strategy is implemented in `systematic_equity_strategy.ipynb`, which provides a complete pipeline including:

- Fama-French 5-Factor alpha estimation
- Technical feature engineering (21 features)
- ML-based alpha prediction (ElasticNet, XGBoost, RandomForest)
- Portfolio optimization using PyPortfolioOpt
- Walk-forward backtesting with transaction costs

**Your task is to replace or augment the traditional ML components with deep learning models and analyze whether they improve strategy performance.**

---

## Dataset

The project uses:
- **S&P 500 prices** (`sp500_prices_with_metadata.parquet`): Daily adjusted close prices from 2000-2020
- **Fama-French 5 Factors** (`FamaFrench_5_Factors_daily_returns.csv`): Market, SMB, HML, RMW, CMA factors
- **Sector metadata** (`sp500_metadata.csv`): GICS sector classifications

**Important**: To reduce computational burden and focus on the most liquid securities, restrict your analysis to the **top 100 largest companies** by market capitalization (or average trading volume) instead of the full S&P 500 universe.

---

## Requirements

You **must** incorporate deep learning in at least one of the following areas:

### 1. Deep Learning Alpha Models

Replace or ensemble with the existing ML models using one of:

#### Option A: Graph Neural Networks (GNN)
- Model stock relationships as a graph (e.g., sector connections, correlation-based edges)
- Use GNN architectures like GAT (Graph Attention Network) or GCN (Graph Convolutional Network)
- Predict cross-sectional alpha rankings
- **Why GNN?** Stock returns exhibit network effects - sector co-movement, supply chain relationships, and correlation clusters that graph structures can capture

#### Option B: Transformer-Based Models
- Treat the stock universe as a sequence/set and use attention mechanisms
- Consider architectures like:
  - Cross-sectional Transformer (attention across stocks at each time point)
  - Temporal Transformer (attention across time for each stock)
  - Hybrid approaches combining both
- **Why Transformers?** Attention mechanisms can learn which stocks/features matter most for predicting returns without explicit feature engineering

#### Option C: Temporal Models
- LSTM/GRU networks for sequential pattern recognition
- Temporal Convolutional Networks (TCN)
- Neural ODE for continuous-time dynamics

### 2. Covariance Matrix Enhancement

Improve the risk model using deep learning:

#### Option A: Diffusion Models for Covariance Denoising
- Train a diffusion model to denoise sample covariance matrices
- The noisy input is the sample covariance; the target is a "cleaner" version (e.g., future realized covariance or shrinkage target)
- Use the denoised covariance in the portfolio optimizer
- **Why Diffusion?** Sample covariance matrices are notoriously noisy, especially with limited data. Diffusion models can learn to remove estimation noise while preserving true correlation structure

#### Option B: Autoencoder-Based Denoising
- Train a variational autoencoder (VAE) on historical covariance matrices
- Use the reconstruction as a denoised estimate
- Alternatively, use the latent space for factor model estimation

---

## Suggested Enhancements

Beyond the required deep learning components, consider:

### Additional Features
- **Alternative data proxies**: Momentum across different horizons, volatility regimes
- **Cross-sectional features**: Relative strength vs. sector, percentile rankings
- **Interaction features**: Factor exposures interacted with market regime indicators
- **Lagged features**: Multi-horizon returns and volatility at various lookbacks
- **Graph-based features**: Centrality measures, clustering coefficients from correlation networks

### Architecture Improvements
- Multi-task learning (predict returns + volatility jointly)
- Adversarial training for robustness
- Ensemble deep learning models with traditional ML
- Attention-based feature importance analysis

### Training Considerations
- Proper temporal cross-validation (no look-ahead bias)
- Regularization techniques (dropout, weight decay, early stopping)
- Data augmentation for financial time series
- Handling class imbalance if predicting directional returns

---

## Deliverables

1. **Enhanced Notebook**: Modified version of `systematic_equity_strategy.ipynb` with your deep learning enhancements

2. **Analysis Report** (within the notebook or separate markdown):
   - Description of your deep learning architecture(s)
   - Training methodology and hyperparameter choices
   - Performance comparison: baseline vs. enhanced strategy
   - Ablation studies (which components helped most?)
   - Discussion of overfitting risks and mitigation strategies

3. **Performance Metrics to Report**:
   - CAGR, Sharpe Ratio, Sortino Ratio
   - Maximum Drawdown, Calmar Ratio
   - Win Rate, Average Turnover
   - Statistical significance of improvements (if any)

---

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Deep Learning Implementation Quality | 30% |
| Methodological Rigor (no look-ahead, proper validation) | 25% |
| Analysis Depth and Insights | 25% |
| Code Quality and Documentation | 10% |
| Creativity and Innovation | 10% |

**Note**: Outperforming the baseline is NOT required for a good grade. A well-implemented model that underperforms but includes thoughtful analysis of why is equally valuable. Financial markets are hard to predict - understanding failure modes is as important as achieving success.

---

## Getting Started

1. **Run the baseline notebook** to understand the existing pipeline
2. **Filter to top 100 stocks** by modifying the data loading section
3. **Choose your deep learning approach** (alpha model, covariance denoising, or both)
4. **Implement incrementally** - start simple, add complexity
5. **Compare rigorously** - use the same backtest period and methodology

### Suggested Libraries
- PyTorch or TensorFlow for deep learning
- PyTorch Geometric for GNN implementations
- Hugging Face Transformers for attention-based architectures
- PyPortfolioOpt (already used in baseline) for optimization

---

## Tips for Success

1. **Start with the baseline**: Make sure you understand every component before modifying
2. **Keep it simple initially**: A well-tuned simple model beats a poorly-tuned complex one
3. **Watch for overfitting**: Financial data is noisy; complex models overfit easily
4. **Document everything**: Your analysis of what works/doesn't work is as valuable as the results
5. **Be skeptical of good results**: If your Sharpe ratio is 3+, you probably have a bug

