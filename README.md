# Econometrics Project: Loan Default Prediction

This document presents **Part 2: Loan Default Prediction**. The analysis includes data preparation, variable creation, and outlier treatment.

## Data Overview

- **Total observations:** 30,000
- **Target variable (y):** 1 = default, 0 = no default
- **Default rate:** 22.12%

---

## Stage 1. Data Preparation

### 1.1 Handling Missing Values

- Missing values in `edu`: 0 observations
- Filled with high school (code 3)

### 1.2 Recoding Education

- Values 4 and 5 in `edu` were recoded to 3 (high school)

### 1.3 Creating New Variables

**Average Payment Delay (avg_delay):**
- Calculated from columns: delay_apr, delay_may, delay_jun, delay_jul, delay_aug, delay_sep
- Range: 0.00 to 6.00 months
- Mean: 0.28 months
- Median: 0.00 months

**Average Debt-to-Limit Ratio (avg_debt_to_limit):**
- Calculated from columns: debt_may, debt_jun, debt_jul, debt_aug, debt_sep
- For each month: debt / limit
- Average of 5 monthly ratios

### 1.4 Outlier Treatment

- Extreme values in `avg_debt_to_limit` were capped at the 99.9th percentile
- Cap value: 1.4422 (144.22%)
- Observations affected: 30 (0.100%)

**Statistics after treatment:**
- Min: -1.5684 (-156.84%)
- Max: 1.4422 (144.22%)
- Mean: 0.3228 (32.28%)
- Median: 0.2152 (21.52%)

---

## Stage 2. Exploratory Data Analysis

### 2.1 Summary Statistics

| Variable | Mean | Median | Std Dev | Min | Max |
|----------|------|--------|---------|-----|-----|
| Age | 35.5 | 34.0 | 9.2 | 21 | 79 |
| avg_delay | 0.28 | 0.00 | 0.60 | 0.00 | 6.00 |
| avg_debt_to_limit | 0.3228 | 0.2152 | 0.3338 | -1.5684 | 1.4422 |

### 2.2 Frequency Tables

| Variable | Category | Count | Percentage | Default Rate |
|----------|----------|-------|------------|--------------|
| Sex | Male | 11,888 | 39.6% | 24.2% |
| Sex | Female | 18,112 | 60.4% | 20.8% |
| Education | Graduate | 10,585 | 35.3% | 19.2% |
| Education | University | 14,030 | 46.8% | 23.7% |
| Education | High school | 5,385 | 17.9% | 23.6% |
| Marital | Unknown | 54 | 0.2% | 9.3% |
| Marital | Married | 13,659 | 45.5% | 23.5% |
| Marital | Single | 15,964 | 53.2% | 20.9% |
| Marital | Other | 323 | 1.1% | 26.0% |

---

## Stage 3. Logistic Regression Results

### 3.1 Model Estimates

| Variable | Coefficient | Odds Ratio | p-value |
|----------|-------------|------------|---------|
| Intercept | -1.7459 | 0.174 | 5.1021e-48 |
| Age | 0.0027 | 1.003 | 0.1775 |
| Sex (female=1) | -0.1274 | 0.880 | 2.3006e-04 |
| Education | 0.0131 | 1.013 | 0.5980 |
| Marital Status | -0.0814 | 0.922 | 2.3171e-02 |
| Avg Payment Delay | 1.3871 | 4.003 | 0.0000e+00 |
| Avg Debt-to-Limit | 0.2439 | 1.276 | 4.0216e-06 |

**Key interpretations:**
- Women have **12% lower** odds of default than men (OR = 0.88)
- One month increase in payment delay **quadruples** odds of default (OR = 4.00)
- 100pp increase in debt-to-limit increases odds by **28%** (OR = 1.28)
- Age and education are **not statistically significant**

### 3.2 Marginal Effect (Debt-to-Limit Ratio)

When `avg_debt_to_limit` increases by 1 (100 percentage points), the **probability of default increases by 4.20 percentage points**, evaluated at the mean predicted probability (p̄ = 0.2212).

---

## Stage 4. In-Sample Model Performance

Model performance is evaluated on the training set (80% of data).

### 4.1 Confusion Matrix (Cutoff = 0.5)

| | | Predicted | |
|----------------|-----------|-----------|-----------|
| | | **No Default** | **Default** |
| **Actual** | **No Default** | 18,038 | 653 |
| | **Default** | 4,046 | 1,263 |

### 4.2 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 0.8042 (80.4%) | Overall correct predictions |
| Sensitivity | 0.2379 (23.8%) | % of actual defaults detected |
| Specificity | 0.9651 (96.5%) | % of actual non-defaults correctly identified |
| AUC | 0.7377 | Ability to distinguish between classes (0.5=random, 1=perfect) |
| Nagelkerke R² | 0.1839 | Pseudo R² — model explains 18.4% of variation |

### 4.3 Accuracy vs. Cutoff

![Accuracy vs Cutoff](figures/accuracy_vs_cutoff.png)

The plot shows overall accuracy across different classification thresholds. Accuracy peaks around cutoff = 0.3-0.5. Lower cutoffs increase sensitivity (catch more defaults) but decrease specificity (more false alarms). The default cutoff of 0.5 is near the optimum.

### 4.4 ROC Curve

![ROC Curve](figures/roc_curve.png)

The ROC curve plots Sensitivity (True Positive Rate) against 1-Specificity (False Positive Rate). The Area Under the Curve (AUC = 0.7377) indicates **good** discriminatory power. A model with no predictive power would follow the diagonal line (AUC = 0.5).

---


---

## Stage 5. Out-of-Sample Prediction

To assess how well the model generalizes to unseen data, we evaluate its performance on the test set (20% of the original data, held out from training).

### 5.1 Confusion Matrix (Cutoff = 0.5)

| | | Predicted | |
|----------------|-----------|-----------|-----------|
| | | **No Default** | **Default** |
| **Actual** | **No Default** | 4,491 | 182 |
| | **Default** | 1,012 | 315 |

### 5.2 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8010 (80.10%) |
| **Sensitivity (Recall)** | 0.2374 (23.74%) |
| **Specificity** | 0.9611 (96.11%) |

### 5.3 Comparison with In-Sample Performance

| Metric | In-Sample (Train) | Out-of-Sample (Test) | Difference |
|--------|-------------------|---------------------|------------|
| Accuracy | 0.8042 | 0.8010 | -0.0032 |
| Sensitivity | 0.2379 | 0.2374 | -0.0005 |
| Specificity | 0.9651 | 0.9611 | -0.0040 |

### 5.4 Interpretation

The out-of-sample performance is **very close** to the in-sample performance. All metrics differ by less than 0.5 percentage points.

**Key findings:**

1. **No overfitting:** The model performs similarly on training and test data
2. **Generalization:** The model successfully captures underlying patterns, not just noise
3. **Consistency:** The conservative nature (high specificity, low sensitivity) remains stable

This indicates the model is robust and reliable for predicting default risk on new clients.

---

## Conclusion

### Key Findings

1. **Strongest predictors:** Average payment delay (OR = 4.00) and debt-to-limit ratio (OR = 1.28)
2. **Demographics:** Women have 12% lower default odds than men; marital status also matters
3. **Model performance:** AUC = 0.74, Nagelkerke R² = 18.4%
4. **No overfitting:** Out-of-sample metrics match training

### Limitations

- Low sensitivity (only 24% of defaults detected at cutoff = 0.5)
- Education and age not significant

### Future Improvements

- Use cutoff = 0.20 to increase sensitivity to 55% (at cost of specificity)
- Convert marital status to dummy variables (improves fit slightly)
- Test more complex models (random forests, XGBoost)

---

## Appendix: Model Improvements

Three modifications were tested to explore potential improvements beyond the baseline model.

### A.1 Dummy Variables

Education and marital status were converted from numeric codes to dummy variables. This allows each category to have its own coefficient rather than assuming a linear relationship.

| Metric | Original | Dummies | Change |
|--------|----------|---------|--------|
| Accuracy | 0.8010 | 0.8020 | +0.0010 |
| Sensitivity | 0.2374 | 0.2411 | +0.0038 |
| AUC | 0.7377 | 0.7385 | +0.0008 |

**Result:** All metrics improved slightly. Marital status shows strong effects: married, single, and other categories all have higher default risk than "unknown" status.

### A.2 Optimal Cutoff

The default classification cutoff of 0.5 assumes equal cost of false positives and false negatives. For a bank, missing a default (false negative) may be more costly than a false alarm. We searched for the cutoff that maximizes F1-score (harmonic mean of precision and recall).

| Cutoff | Accuracy | Sensitivity | Specificity |
|--------|----------|-------------|-------------|
| 0.50 (default) | 0.8010 | 0.2374 | 0.9611 |
| 0.20 (optimal) | 0.7573 | 0.5463 | 0.8172 |

**Result:** Sensitivity jumps from 23.7% to **54.6%**, meaning the model catches more than twice as many actual defaults. The trade-off is lower specificity (81.7% vs 96.1%).

### A.3 Interaction Term

An interaction term `avg_delay × avg_debt_to_limit` was added to test whether the effect of payment delay depends on the debt level.

- **Coefficient:** -0.6922 (p-value = 0.0000)
- **Interpretation:** The interaction is statistically significant and **negative**. This means the effect of payment delay **weakens** as debt-to-limit ratio increases. In other words, for already highly indebted clients, additional payment delays matter less.

| Metric | Original | Interaction | Change |
|--------|----------|-------------|--------|
| Accuracy | 0.8010 | 0.8028 | +0.0018 |
| Sensitivity | 0.2374 | 0.2411 | +0.0038 |
| AUC | 0.7377 | 0.7333 | -0.0044 |

### A.4 Summary and Recommendation

| Modification | Best For | Key Trade-off |
|--------------|----------|---------------|
| Dummy variables | Slight overall improvement | Minimal downside |
| Cutoff = 0.20 | Catching more defaults | More false alarms |
| Interaction term | Understanding relationships | No AUC improvement |

**Recommendation:** If the bank prioritizes identifying risky clients (even with some false alarms), use **cutoff = 0.20**. If interpretability matters most, stick with the original model. Dummy variables for marital status can be added for a small but consistent improvement.
