# Week 4: Conditional GANs for Bond Pricing

**Deep Learning in Finance - Baruch MFE 2025**

---

## Overview

This week focuses on using **Conditional Generative Adversarial Networks (cGANs)** for bond pricing, with emphasis on:
- Distributional predictions (full price distributions, not just point estimates)
- Uncertainty quantification (calibrated prediction intervals)
- Correlated scenario generation for portfolio risk analysis
- VaR/CVaR calculations and stress testing

---

## Learning Path

### **IMPORTANT: Read the notebooks in order before starting the homework!**

The homework assumes you understand the data preparation, baseline modeling, and cGAN concepts from the first three notebooks.

---

## Part 1: Understanding the Data

### 📓 **Notebook 1: `01_trace_data_preparation.ipynb`**

**Purpose**: Prepare TRACE bond transaction data merged with bond characteristics

**What you'll learn:**
- **TRACE data**: Trade Reporting and Compliance Engine - contains actual bond transactions
- **Data sources**:
  - `hy_traces_2025.csv` - Transaction-level data (210 MB, ~50K trades)
  - `hy_universe.csv` - Bond metadata (990 KB, 660 bonds)
- **Data cleaning**:
  - Merge trade data with bond characteristics (sector, rating, maturity, etc.)
  - Handle missing values using sector/rating medians
  - Calculate daily volume-weighted average prices (VWAP)
  - Filter for liquid bonds (minimum trade frequency)
- **Date range**: January 2025 - August 2025 (160 trading days)
- **Output**: `trace_hy_merged.csv` - Clean dataset ready for modeling

**Key takeaways:**
- Bond prices come from actual trades, not quotes
- Data quality matters: we filter for liquidity and clean outliers
- Bond characteristics (sector, rating, maturity) are crucial features

**Time to read**: ~15 minutes

---

## Part 2: Baseline Model

### 📓 **Notebook 2: `02_hy_daily_pricing_model.ipynb`**

**Purpose**: Build ElasticNet baseline for bond pricing (traditional regression)

**What you'll learn:**
- **Feature engineering**:
  - Time-varying features: years to maturity, time since issuance
  - Categorical encoding: sector, rating, seniority
  - Binary flags: putable, callable, convertible
- **Modeling approach**:
  - Day 1: Price level model (predict absolute price)
  - Days 2+: Price change model (predict ΔPrice with lagged price)
- **ElasticNet**: Linear regression with L1+L2 regularization
  - Automatically selects important features
  - Prevents overfitting with regularization
- **Performance**: ~7-10 RMSE on daily predictions
- **Limitations**:
  - ✅ Fast training (<1 minute)
  - ✅ Interpretable coefficients
  - ❌ Only point predictions (no uncertainty)
  - ❌ No correlation modeling

**Key takeaways:**
- ElasticNet provides a strong baseline for comparison
- Feature engineering is critical for performance
- Traditional models can't quantify uncertainty

**Time to read**: ~20 minutes

---

## Part 3: Conditional GAN Illustration

### 📓 **Notebook 3: `03_cgan_bond_pricing.ipynb`**

**Purpose**: Demonstrate cGAN for distributional bond pricing with scenario analysis

**What you'll learn:**

#### **Architecture:**
- **Generator**: G(z, features) → price
  - Input: Random noise z + bond features
  - Output: Predicted price
  - Purpose: Generate realistic prices given bond characteristics
- **Discriminator**: D(features, price) → [0, 1]
  - Input: Bond features + price (real or fake)
  - Output: Probability that (features, price) pair is real
  - Purpose: Force generator to learn realistic price distributions

#### **Training:**
- Train/test split: First 22 days (training), next 22 days (testing)
- Adversarial training: Generator tries to fool Discriminator
- 2000 epochs, ~5-10 minutes on CPU
- No data leakage: strict chronological split

#### **Key Innovation - Distributional Predictions:**
Instead of one price prediction, cGAN generates 1000 samples:
```
Bond A: [95.2, 94.8, 96.1, ..., 95.5]  ← 1000 samples
        ↓
Mean:   95.3
5th %:  92.1  ← 90% prediction interval
95th%:  98.5  ←
```

#### **Scenario Generation:**
The most powerful feature - generate 1000 correlated market scenarios:
- **Same noise z** for all bonds → realistic correlations
- Each scenario = one possible future with prices for ALL 660 bonds
- Enables portfolio VaR, CVaR, stress testing

#### **Results:**
- **Point accuracy**: cGAN ≈ ElasticNet (~6-10 RMSE)
- **Calibration**: 90% intervals contain ~90% of actual prices
- **Correlations**: cGAN captures realistic sector co-movements
- **Portfolio risk**: More accurate VaR/CVaR than parametric models

**Key takeaways:**
- cGAN models P(Price|Features), not just E[Price|Features]
- Calibrated intervals are crucial for risk management
- Correlated scenarios enable proper portfolio analysis
- Trade-off: 5-10 min training vs <1 min for ElasticNet

**Time to read**: ~45 minutes (most important notebook!)

---

## Part 4: Your Homework

### 📝 **Notebook 4: `04_cgan_bond_pricing_homework.ipynb`**

**Now you're ready for the homework!**

**What you'll do:**
- ✅ All model training code is provided (just run it)
- ✅ Your task: Complete analysis questions and interpretations
- ✅ Focus areas:
  1. Compare cGAN vs ElasticNet point predictions
  2. Calculate and interpret prediction intervals (calibration)
  3. Generate 1000 correlated scenarios for portfolio analysis
  4. Compute VaR and CVaR for risk management
  5. Analyze sector correlations and diversification benefits

**Grading:**
- Section 1 (Model Comparison): 20 points
- Section 2 (Distributional Predictions): 25 points
- Section 3 (Portfolio Risk Analysis): 30 points
- Section 4 (Correlation & Diversification): 25 points
- **Total**: 100 points

**Time to complete**: ~3-4 hours

---

## File Structure

```
week4/
├── README.md                                          ← You are here
├── hy_universe.csv                                    ← Bond metadata (990 KB)
├── hy_traces_2025.csv                                 ← Transaction data (210 MB)
├── 01_trace_data_preparation.ipynb                    ← Read first
├── 02_hy_daily_pricing_model.ipynb                    ← Read second
├── 03_cgan_bond_pricing.ipynb                         ← Read third (most important!)
├── homework04_bond_pricing_cgan.ipynb                 ← Complete this for submission
```

---

## Data Files

### **Input Data (Provided):**

1. **`hy_universe.csv`** (990 KB)
   - 660 unique high-yield bonds
   - Columns: cusip_9, issuer, sector, rating, maturity, coupon, seniority, etc.
   - Source: Bond characteristics from offering documents

2. **`hy_traces_2025.csv`** (210 MB)
   - ~50,000 bond transactions
   - Columns: cusip_9, date, price, volume, side (buy/sell)
   - Source: TRACE (Trade Reporting and Compliance Engine)
   - Date range: January - August 2025

### **Generated Data (Created by notebooks):**

3. **`trace_hy_merged.csv`**
   - Created by Notebook 1
   - Merged transaction + bond characteristics
   - Daily VWAP prices for 660 bonds over 160 days

---

## Prerequisites

### **Python Packages:**
```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn
```

### **Knowledge:**
- Basic understanding of neural networks
- Familiarity with regression models
- Concepts: VaR, CVaR, correlation, diversification
- (All concepts will be explained in notebooks)

---

## Getting Started

### **Step-by-step instructions:**

1. **Ensure you have the data files:**
   ```bash
   ls hy_universe.csv hy_traces_2025.csv
   ```

2. **Read Notebook 1** (Data Preparation):
   - Open `01_trace_data_preparation.ipynb`
   - Run all cells to understand data cleaning
   - Verify `trace_hy_merged.csv` is created

3. **Read Notebook 2** (ElasticNet Baseline):
   - Open `02_hy_daily_pricing_model.ipynb`
   - Understand feature engineering and baseline model
   - Note the limitations (no uncertainty quantification)

4. **Read Notebook 3** (Conditional GAN):
   - Open `03_cgan_bond_pricing.ipynb`
   - **This is the most important notebook!**
   - Understand: architecture, training, distributional predictions, scenario generation
   - Pay special attention to the "Scenario Analysis" section

5. **Complete the Homework**:
   - Open `04_cgan_bond_pricing_homework.ipynb`
   - Run all cells in Part A (provided code)
   - Complete all questions in Part B
   - Fill in TODO sections and interpretation questions

6. **Submit**:
   - Export notebook as PDF or HTML
   - Upload to Coursera
   - Check submission checklist in notebook

---

## Tips for Success

### **Understanding the notebooks:**
- 📖 **Don't just run cells** - read the markdown explanations carefully
- 🤔 **Ask yourself**: Why is this step necessary? What would happen without it?
- 📊 **Study the plots**: What patterns do you see? What do they tell you?
- 🔗 **Connect concepts**: How does Notebook 2 relate to Notebook 3?

### **For the homework:**
- ✅ **Read all three notebooks first** - don't start homework blind!
- ✅ **Understand, don't memorize** - homework tests comprehension, not code copying
- ✅ **Think practically** - how would you use this in a real trading desk?
- ✅ **Check your answers** - do the numbers make sense? Are interpretations clear?

### **Common pitfalls to avoid:**
- ❌ Skipping the data preparation notebook (you won't understand the data!)
- ❌ Not understanding why correlations matter (critical for portfolio risk)
- ❌ Confusing VaR with CVaR (they measure different things!)
- ❌ Writing vague interpretations ("the model is good") - be specific!

---

## Learning Objectives

By the end of this week, you should be able to:

✅ **Explain** the difference between point predictions and distributional predictions
✅ **Implement** conditional GANs for financial time series (code provided, but understand it!)
✅ **Calculate** calibrated prediction intervals and assess model calibration
✅ **Generate** correlated market scenarios using the same noise vector
✅ **Compute** portfolio VaR and CVaR from scenario distributions
✅ **Analyze** sector correlations and quantify diversification benefits
✅ **Articulate** when to use cGAN vs traditional regression in practice
✅ **Connect** academic concepts to real-world risk management applications

---

## Key Concepts to Master

### **From Notebook 1 (Data):**
- TRACE transaction data vs bond characteristics
- Volume-weighted average price (VWAP)
- Data cleaning: outliers, missing values, liquidity filters

### **From Notebook 2 (Baseline):**
- Feature engineering for bond pricing
- ElasticNet regularization (L1 + L2)
- Price level models vs price change models

### **From Notebook 3 (cGAN):**
- Conditional GAN architecture (Generator + Discriminator)
- Distributional predictions: P(Price|Features)
- Calibration: empirical coverage vs expected coverage
- Scenario generation: using same noise vector for correlation
- Portfolio risk: VaR, CVaR, tail risk

### **For Homework:**
- Model comparison: when is cGAN better than regression?
- Uncertainty quantification: interpreting prediction intervals
- Correlation analysis: sector co-movements
- Diversification: quantifying risk reduction

---

## Questions?

If you have questions:
1. **First**: Re-read the relevant notebook carefully
2. **Second**: Check if your question is answered in this README
3. **Third**: Discuss with classmates (collaboration encouraged for understanding!)
4. **Fourth**: Post on course forum or reach out to instructor

---

## Important Notes


### **Grading Criteria:**
- **Correctness** (50%): Code runs, calculations are accurate
- **Interpretation** (40%): Answers show deep understanding and practical insight
- **Presentation** (10%): Clear writing, well-formatted code, organized output

---

## Summary

**Before starting homework:**
1. ✅ Read `01_trace_data_preparation.ipynb` - understand the data
2. ✅ Read `02_hy_daily_pricing_model.ipynb` - understand the baseline
3. ✅ Read `03_cgan_bond_pricing.ipynb` - understand cGAN capabilities
4. ✅ Review this README for overview and tips

**Then:**
5. 📝 Complete `homework04_bond_pricing_cgan.ipynb`
6. 🎯 Submit for grading

---

**Good luck! This is one of the most practical and impactful weeks of the course.**

The concepts you learn here - distributional predictions, uncertainty quantification, scenario analysis, and portfolio risk management - are directly applicable to careers in quantitative finance, risk management, and algorithmic trading.

---

*Last updated: 2025-01-24*
