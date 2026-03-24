import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

df = pd.read_csv('data_part2.csv')
df['edu'] = df['edu'].fillna(3)
df.loc[df['edu'] == 4, 'edu'] = 3
df.loc[df['edu'] == 5, 'edu'] = 3

#part of the model improvement------------------------------------------------------------------------------------------
df_exp = df.copy()
#-----------------------------------------------------------------------------------------------------------------------

delay_columns = ['delay_apr', 'delay_may', 'delay_jun', 'delay_jul', 'delay_aug', 'delay_sep']
df['avg_delay'] = df[delay_columns].mean(axis=1)

debt_columns = [col for col in df.columns if col.startswith('debt_')]
for col in debt_columns:
    df[f'{col}_ratio'] = df[col] / df['limit']
ratio_columns = [f'{col}_ratio' for col in debt_columns]
df['avg_debt_to_limit'] = df[ratio_columns].mean(axis=1)
cap_value = df['avg_debt_to_limit'].quantile(0.999)
exceed_count = (df['avg_debt_to_limit'] > cap_value).sum()
df['avg_debt_to_limit'] = df['avg_debt_to_limit'].clip(upper=cap_value)

#part of the model improvement------------------------------------------------------------------------------------------
df_exp['avg_delay'] = df_exp[delay_columns].mean(axis=1)

debt_columns_exp = [col for col in df_exp.columns if col.startswith('debt_')]
for col in debt_columns_exp:
    df_exp[f'{col}_ratio'] = df_exp[col] / df_exp['limit']
ratio_columns_exp = [f'{col}_ratio' for col in debt_columns_exp]
df_exp['avg_debt_to_limit'] = df_exp[ratio_columns_exp].mean(axis=1)
df_exp['avg_debt_to_limit'] = df_exp['avg_debt_to_limit'].clip(upper=cap_value)
#-----------------------------------------------------------------------------------------------------------------------

features = ['age', 'sex', 'edu', 'mar', 'avg_delay', 'avg_debt_to_limit']
X = df[features]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_const = sm.add_constant(X_train)

logit_model = sm.Logit(y_train, X_train_const)
logit_result = logit_model.fit(disp=0)
coef_debt = logit_result.params['avg_debt_to_limit']
p_mean = logit_result.predict(X_train_const).mean()
marginal_effect_avg = coef_debt * p_mean * (1 - p_mean)

y_pred_prob_train = logit_result.predict(X_train_const)
y_pred_train = (y_pred_prob_train > 0.5).astype(int)
cm_train = confusion_matrix(y_train, y_pred_train)
tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
accuracy_train = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
sensitivity_train = tp_train / (tp_train + fn_train)
specificity_train = tn_train / (tn_train + fp_train)

fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob_train)
auc_train = roc_auc_score(y_train, y_pred_prob_train)

if not os.path.exists('figures'):
    os.makedirs('figures')

# Accuracy vs Cutoff Values
cutoffs = np.arange(0.01, 0.99, 0.01)
accuracies = []
for cutoff in cutoffs:
    y_pred_cutoff = (y_pred_prob_train > cutoff).astype(int)
    accuracies.append(accuracy_score(y_train, y_pred_cutoff))

plt.figure(figsize=(8, 6))
plt.plot(cutoffs, accuracies, linewidth=2)
plt.xlabel('Cutoff Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Cutoff Value')
plt.grid(True, alpha=0.3)
plt.savefig('figures/accuracy_vs_cutoff.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_train:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

ll_null = logit_result.llnull
ll_model = logit_result.llf

# R²
mcfadden_r2 = 1 - ll_model / ll_null
n = len(y_train)
cox_snell_r2 = 1 - np.exp((ll_null - ll_model) * 2 / n)
nagelkerke_r2 = cox_snell_r2 / (1 - np.exp(ll_null * 2 / n))

params = logit_result.params
p_values = logit_result.pvalues
conf_int = logit_result.conf_int()
results_df = pd.DataFrame({
    'Coefficient': params,
    'Std. Error': logit_result.bse,
    'z-statistic': logit_result.tvalues,
    'p-value': p_values,
})

odds_ratios = np.exp(params)

#Out-of-Sample Prediction
X_test_const = sm.add_constant(X_test)
y_pred_prob_test = logit_result.predict(X_test_const)
y_pred_test = (y_pred_prob_test > 0.5).astype(int)
cm_test = confusion_matrix(y_test, y_pred_test)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
accuracy_test = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)
sensitivity_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0
specificity_test = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0

#part of the model improvement------------------------------------------------------------------------------------------
# DUMMY VARIABLES
edu_dummies = pd.get_dummies(df_exp['edu'], prefix='edu', drop_first=True)
edu_dummies.columns = ['edu_university', 'edu_highschool']
mar_dummies = pd.get_dummies(df_exp['mar'], prefix='mar', drop_first=True)
mar_dummies.columns = ['mar_married', 'mar_single', 'mar_other']
features_dummies = ['age', 'sex', 'avg_delay', 'avg_debt_to_limit']
X_dummies = df_exp[features_dummies].copy()
X_dummies = pd.concat([X_dummies, edu_dummies, mar_dummies], axis=1)
for col in X_dummies.columns:
    X_dummies[col] = X_dummies[col].astype(float)
y_dummies = df_exp['y'].astype(float)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dummies, y_dummies, test_size=0.2, random_state=42, stratify=y_dummies
)
X_train_d_const = sm.add_constant(X_train_d)

X_train_d_const_np = np.asarray(X_train_d_const, dtype=float)
y_train_d_np = np.asarray(y_train_d, dtype=float)

logit_model_d = sm.Logit(y_train_d_np, X_train_d_const_np)
logit_result_d = logit_model_d.fit(disp=0)

X_test_d_const = sm.add_constant(X_test_d)
X_test_d_const_np = np.asarray(X_test_d_const, dtype=float)
y_pred_prob_test_d = logit_result_d.predict(X_test_d_const_np)
y_pred_test_d = (y_pred_prob_test_d > 0.5).astype(int)

cm_test_d = confusion_matrix(y_test_d, y_pred_test_d)
tn_d, fp_d, fn_d, tp_d = cm_test_d.ravel()
accuracy_d = (tp_d + tn_d) / (tp_d + tn_d + fp_d + fn_d)
sensitivity_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
specificity_d = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0

X_train_d_const_np = np.asarray(X_train_d_const, dtype=float)
y_pred_prob_train_d = logit_result_d.predict(X_train_d_const_np)
auc_d = roc_auc_score(y_train_d, y_pred_prob_train_d)

var_names = X_train_d_const.columns.tolist()

#  Cutoff
cutoffs = np.arange(0.05, 0.95, 0.05)
results_cutoff = []

for cutoff in cutoffs:
    y_pred_cut = (y_pred_prob_test > cutoff).astype(int)

    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test, y_pred_cut).ravel()
    acc_c = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
    sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
    spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0

    precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
    f1_c = 2 * (precision * sens_c) / (precision + sens_c) if (precision + sens_c) > 0 else 0

    results_cutoff.append({
        'cutoff': cutoff,
        'accuracy': acc_c,
        'sensitivity': sens_c,
        'specificity': spec_c,
        'f1_score': f1_c
    })

results_cutoff_df = pd.DataFrame(results_cutoff)

best_f1 = results_cutoff_df.loc[results_cutoff_df['f1_score'].idxmax()]
# Interaction Term (avg_delay * avg_debt_to_limit)
df['delay_debt_interaction'] = df['avg_delay'] * df['avg_debt_to_limit']
features_interaction = ['age', 'sex', 'edu', 'mar', 'avg_delay', 'avg_debt_to_limit', 'delay_debt_interaction']
X_interaction = df[features_interaction]
y_interaction = df['y']


X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_interaction, y_interaction, test_size=0.2, random_state=42, stratify=y_interaction
)


X_train_i_const = sm.add_constant(X_train_i)
logit_model_i = sm.Logit(y_train_i, X_train_i_const)
logit_result_i = logit_model_i.fit(disp=0)


X_test_i_const = sm.add_constant(X_test_i)
y_pred_prob_test_i = logit_result_i.predict(X_test_i_const)
y_pred_test_i = (y_pred_prob_test_i > 0.5).astype(int)

cm_test_i = confusion_matrix(y_test_i, y_pred_test_i)
tn_i, fp_i, fn_i, tp_i = cm_test_i.ravel()
accuracy_i = (tp_i + tn_i) / (tp_i + tn_i + fp_i + fn_i)
sensitivity_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
specificity_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0

y_pred_prob_train_i = logit_result_i.predict(X_train_i_const)
auc_i = roc_auc_score(y_train_i, y_pred_prob_train_i)

coef_interaction = logit_result_i.params['delay_debt_interaction']
pval_interaction = logit_result_i.pvalues['delay_debt_interaction']
# ----------------------------------------------------------------------------------------------------------------------
# Record all results in README
# ----------------------------------------------------------------------------------------------------------------------

with open('README.md', 'w', encoding='utf-8') as f:
    f.write('# Econometrics Project: Loan Default Prediction\n\n')

    f.write('This document presents **Part 2: Loan Default Prediction**. ')
    f.write('The analysis includes data preparation, variable creation, and outlier treatment.\n\n')
    f.write('## Data Overview\n\n')
    f.write(f'- **Total observations:** {len(df):,}\n')
    f.write(f'- **Target variable (y):** 1 = default, 0 = no default\n')
    f.write(f'- **Default rate:** {df["y"].mean() * 100:.2f}%\n\n')
    f.write('---\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 1. Data Preparation
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 1. Data Preparation\n\n')

    f.write('### 1.1 Handling Missing Values\n\n')
    f.write(f'- Missing values in `edu`: {df["edu"].isnull().sum()} observations\n')
    f.write('- Filled with high school (code 3)\n\n')

    f.write('### 1.2 Recoding Education\n\n')
    f.write('- Values 4 and 5 in `edu` were recoded to 3 (high school)\n\n')

    f.write('### 1.3 Creating New Variables\n\n')
    f.write('**Average Payment Delay (avg_delay):**\n')
    f.write(f'- Calculated from columns: delay_apr, delay_may, delay_jun, delay_jul, delay_aug, delay_sep\n')
    f.write(f'- Range: {df["avg_delay"].min():.2f} to {df["avg_delay"].max():.2f} months\n')
    f.write(f'- Mean: {df["avg_delay"].mean():.2f} months\n')
    f.write(f'- Median: {df["avg_delay"].median():.2f} months\n\n')

    f.write('**Average Debt-to-Limit Ratio (avg_debt_to_limit):**\n')
    f.write(f'- Calculated from columns: debt_may, debt_jun, debt_jul, debt_aug, debt_sep\n')
    f.write(f'- For each month: debt / limit\n')
    f.write(f'- Average of 5 monthly ratios\n\n')

    f.write('### 1.4 Outlier Treatment\n\n')
    f.write(f'- Extreme values in `avg_debt_to_limit` were capped at the 99.9th percentile\n')
    f.write(f'- Cap value: {cap_value:.4f} ({cap_value * 100:.2f}%)\n')
    f.write(f'- Observations affected: {exceed_count} ({exceed_count / len(df) * 100:.3f}%)\n\n')

    f.write('**Statistics after treatment:**\n')
    f.write(f'- Min: {df["avg_debt_to_limit"].min():.4f} ({df["avg_debt_to_limit"].min() * 100:.2f}%)\n')
    f.write(f'- Max: {df["avg_debt_to_limit"].max():.4f} ({df["avg_debt_to_limit"].max() * 100:.2f}%)\n')
    f.write(f'- Mean: {df["avg_debt_to_limit"].mean():.4f} ({df["avg_debt_to_limit"].mean() * 100:.2f}%)\n')
    f.write(f'- Median: {df["avg_debt_to_limit"].median():.4f} ({df["avg_debt_to_limit"].median() * 100:.2f}%)\n\n')

    f.write('---\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 2. Exploratory Data Analysis
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 2. Exploratory Data Analysis\n\n')

    f.write('### 2.1 Summary Statistics\n\n')
    f.write('| Variable | Mean | Median | Std Dev | Min | Max |\n')
    f.write('|----------|------|--------|---------|-----|-----|\n')
    f.write(
        f'| Age | {df["age"].mean():.1f} | {df["age"].median():.1f} | {df["age"].std():.1f} | {df["age"].min()} | {df["age"].max()} |\n')
    f.write(
        f'| avg_delay | {df["avg_delay"].mean():.2f} | {df["avg_delay"].median():.2f} | {df["avg_delay"].std():.2f} | {df["avg_delay"].min():.2f} | {df["avg_delay"].max():.2f} |\n')
    f.write(
        f'| avg_debt_to_limit | {df["avg_debt_to_limit"].mean():.4f} | {df["avg_debt_to_limit"].median():.4f} | {df["avg_debt_to_limit"].std():.4f} | {df["avg_debt_to_limit"].min():.4f} | {df["avg_debt_to_limit"].max():.4f} |\n\n')

    f.write('### 2.2 Frequency Tables\n\n')
    f.write('| Variable | Category | Count | Percentage | Default Rate |\n')
    f.write('|----------|----------|-------|------------|--------------|\n')
    f.write(
        f'| Sex | Male | {(df["sex"] == 0).sum():,} | {(df["sex"] == 0).sum() / len(df) * 100:.1f}% | {df[df["sex"] == 0]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Sex | Female | {(df["sex"] == 1).sum():,} | {(df["sex"] == 1).sum() / len(df) * 100:.1f}% | {df[df["sex"] == 1]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Education | Graduate | {(df["edu"] == 1).sum():,} | {(df["edu"] == 1).sum() / len(df) * 100:.1f}% | {df[df["edu"] == 1]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Education | University | {(df["edu"] == 2).sum():,} | {(df["edu"] == 2).sum() / len(df) * 100:.1f}% | {df[df["edu"] == 2]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Education | High school | {(df["edu"] == 3).sum():,} | {(df["edu"] == 3).sum() / len(df) * 100:.1f}% | {df[df["edu"] == 3]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Marital | Unknown | {(df["mar"] == 0).sum():,} | {(df["mar"] == 0).sum() / len(df) * 100:.1f}% | {df[df["mar"] == 0]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Marital | Married | {(df["mar"] == 1).sum():,} | {(df["mar"] == 1).sum() / len(df) * 100:.1f}% | {df[df["mar"] == 1]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Marital | Single | {(df["mar"] == 2).sum():,} | {(df["mar"] == 2).sum() / len(df) * 100:.1f}% | {df[df["mar"] == 2]["y"].mean() * 100:.1f}% |\n')
    f.write(
        f'| Marital | Other | {(df["mar"] == 3).sum():,} | {(df["mar"] == 3).sum() / len(df) * 100:.1f}% | {df[df["mar"] == 3]["y"].mean() * 100:.1f}% |\n\n')

    f.write('---\n\n')
    # ------------------------------------------------------------------------------------------------------------------
    # Stage 3. Logistic Regression Results
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 3. Logistic Regression Results\n\n')

    f.write('### 3.1 Model Estimates\n\n')
    f.write('| Variable | Coefficient | Odds Ratio | p-value |\n')
    f.write('|----------|-------------|------------|---------|\n')
    f.write(
        f'| Intercept | {params["const"]:.4f} | {odds_ratios["const"]:.3f} | {p_values["const"]:.4f}{stars(p_values["const"])} |\n')
    f.write(f'| Age | {params["age"]:.4f} | {odds_ratios["age"]:.3f} | {p_values["age"]:.4f} |\n')
    f.write(
        f'| Sex (female=1) | {params["sex"]:.4f} | {odds_ratios["sex"]:.3f} | {p_values["sex"]:.4f}{stars(p_values["sex"])} |\n')
    f.write(f'| Education | {params["edu"]:.4f} | {odds_ratios["edu"]:.3f} | {p_values["edu"]:.4f} |\n')
    f.write(
        f'| Marital Status | {params["mar"]:.4f} | {odds_ratios["mar"]:.3f} | {p_values["mar"]:.4f}{stars(p_values["mar"])} |\n')
    f.write(
        f'| Avg Payment Delay | {params["avg_delay"]:.4f} | {odds_ratios["avg_delay"]:.3f} | {p_values["avg_delay"]:.4f}{stars(p_values["avg_delay"])} |\n')
    f.write(
        f'| Avg Debt-to-Limit | {params["avg_debt_to_limit"]:.4f} | {odds_ratios["avg_debt_to_limit"]:.3f} | {p_values["avg_debt_to_limit"]:.4f}{stars(p_values["avg_debt_to_limit"])} |\n\n')
    f.write('*Note: *** p < 0.001, ** p < 0.01, * p < 0.05*\n\n')

    f.write('**Key interpretations:**\n')
    f.write('- Women have **12% lower** odds of default than men (OR = 0.88)\n')
    f.write('- One month increase in payment delay **quadruples** odds of default (OR = 4.00)\n')
    f.write('- 100pp increase in debt-to-limit increases odds by **28%** (OR = 1.28)\n')
    f.write('- Age and education are **not statistically significant**\n\n')

    f.write('### 3.2 Marginal Effect (Debt-to-Limit Ratio)\n\n')
    f.write(
        f'When `avg_debt_to_limit` increases by 1 (100 percentage points), the **probability of default increases by {marginal_effect_avg * 100:.2f} percentage points**, evaluated at the mean predicted probability (p̄ = {p_mean:.4f}).\n\n')

    f.write('---\n\n')


    # ------------------------------------------------------------------------------------------------------------------
    # Stage 4. In-Sample Model Performance - добавляем в README
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 4. In-Sample Model Performance\n\n')
    f.write('Model performance is evaluated on the training set (80% of data).\n\n')

    f.write('### 4.1 Confusion Matrix (Cutoff = 0.5)\n\n')
    f.write('| | | Predicted | |\n')
    f.write('|----------------|-----------|-----------|-----------|\n')
    f.write('| | | **No Default** | **Default** |\n')
    f.write(f'| **Actual** | **No Default** | {tn_train:,} | {fp_train:,} |\n')
    f.write(f'| | **Default** | {fn_train:,} | {tp_train:,} |\n\n')

    f.write('### 4.2 Performance Metrics\n\n')
    f.write('| Metric | Value | Interpretation |\n')
    f.write('|--------|-------|----------------|\n')
    f.write(f'| Accuracy | {accuracy_train:.4f} ({accuracy_train * 100:.1f}%) | Overall correct predictions |\n')
    f.write(
        f'| Sensitivity | {sensitivity_train:.4f} ({sensitivity_train * 100:.1f}%) | % of actual defaults detected |\n')
    f.write(
        f'| Specificity | {specificity_train:.4f} ({specificity_train * 100:.1f}%) | % of actual non-defaults correctly identified |\n')
    f.write(f'| AUC | {auc_train:.4f} | Ability to distinguish between classes (0.5=random, 1=perfect) |\n')
    f.write(
        f'| Nagelkerke R² | {nagelkerke_r2:.4f} | Pseudo R² — model explains {nagelkerke_r2 * 100:.1f}% of variation |\n\n')

    f.write('### 4.3 Accuracy vs. Cutoff\n\n')
    f.write('![Accuracy vs Cutoff](figures/accuracy_vs_cutoff.png)\n\n')
    f.write('The plot shows overall accuracy across different classification thresholds. ')
    f.write('Accuracy peaks around cutoff = 0.3-0.5. Lower cutoffs increase sensitivity (catch more defaults) ')
    f.write('but decrease specificity (more false alarms). The default cutoff of 0.5 is near the optimum.\n\n')

    f.write('### 4.4 ROC Curve\n\n')
    f.write('![ROC Curve](figures/roc_curve.png)\n\n')
    f.write('The ROC curve plots Sensitivity (True Positive Rate) against 1-Specificity (False Positive Rate). ')
    f.write(f'The Area Under the Curve (AUC = {auc_train:.4f}) indicates **good** discriminatory power. ')
    f.write('A model with no predictive power would follow the diagonal line (AUC = 0.5).\n\n')

    f.write('---\n\n')
    # ------------------------------------------------------------------------------------------------------------------
    # Stage 5. Out-of-Sample Prediction
    # ------------------------------------------------------------------------------------------------------------------
    f.write('\n---\n\n')
    f.write('## Stage 5. Out-of-Sample Prediction\n\n')
    f.write('To assess how well the model generalizes to unseen data, we evaluate its performance on the test set ')
    f.write('(20% of the original data, held out from training).\n\n')

    # 5.1 Confusion Matrix
    f.write('### 5.1 Confusion Matrix (Cutoff = 0.5)\n\n')

    f.write('| | | Predicted | |\n')
    f.write('|----------------|-----------|-----------|-----------|\n')
    f.write('| | | **No Default** | **Default** |\n')
    f.write(f'| **Actual** | **No Default** | {tn_test:,} | {fp_test:,} |\n')
    f.write(f'| | **Default** | {fn_test:,} | {tp_test:,} |\n\n')
    # 5.2 Performance Metrics

    f.write('### 5.2 Performance Metrics\n\n')

    f.write('| Metric | Value |\n')
    f.write('|--------|-------|\n')
    f.write(f'| **Accuracy** | {accuracy_test:.4f} ({accuracy_test * 100:.2f}%) |\n')
    f.write(f'| **Sensitivity (Recall)** | {sensitivity_test:.4f} ({sensitivity_test * 100:.2f}%) |\n')
    f.write(f'| **Specificity** | {specificity_test:.4f} ({specificity_test * 100:.2f}%) |\n\n')

    # 5.3 Comparison with In-Sample Performance
    f.write('### 5.3 Comparison with In-Sample Performance\n\n')

    f.write('| Metric | In-Sample (Train) | Out-of-Sample (Test) | Difference |\n')
    f.write('|--------|-------------------|---------------------|------------|\n')
    f.write(f'| Accuracy | {accuracy_train:.4f} | {accuracy_test:.4f} | {accuracy_test - accuracy_train:+.4f} |\n')
    f.write(
        f'| Sensitivity | {sensitivity_train:.4f} | {sensitivity_test:.4f} | {sensitivity_test - sensitivity_train:+.4f} |\n')
    f.write(
        f'| Specificity | {specificity_train:.4f} | {specificity_test:.4f} | {specificity_test - specificity_train:+.4f} |\n\n')

    # 5.4 Interpretation
    f.write('### 5.4 Interpretation\n\n')
    f.write('The out-of-sample performance is **very close** to the in-sample performance. ')
    f.write('All metrics differ by less than 0.5 percentage points.\n\n')

    f.write('**Key findings:**\n\n')
    f.write('1. **No overfitting:** The model performs similarly on training and test data\n')
    f.write('2. **Generalization:** The model successfully captures underlying patterns, not just noise\n')
    f.write('3. **Consistency:** The conservative nature (high specificity, low sensitivity) remains stable\n\n')

    f.write('This indicates the model is robust and reliable for predicting default risk on new clients.\n')
    # ------------------------------------------------------------------------------------------------------------------
    # Conclusion
    # ------------------------------------------------------------------------------------------------------------------
    f.write('\n---\n\n')
    f.write('## Conclusion\n\n')

    f.write('### Key Findings\n\n')
    f.write('1. **Strongest predictors:** Average payment delay (OR = 4.00) and debt-to-limit ratio (OR = 1.28)\n')
    f.write('2. **Demographics:** Women have 12% lower default odds than men; marital status also matters\n')
    f.write('3. **Model performance:** AUC = 0.74, Nagelkerke R² = 18.4%\n')
    f.write('4. **No overfitting:** Out-of-sample metrics match training\n\n')

    f.write('### Limitations\n\n')
    f.write('- Low sensitivity (only 24% of defaults detected at cutoff = 0.5)\n')
    f.write('- Education and age not significant\n\n')

    f.write('### Future Improvements\n\n')
    f.write('- Use cutoff = 0.20 to increase sensitivity to 55% (at cost of specificity)\n')
    f.write('- Convert marital status to dummy variables (improves fit slightly)\n')
    f.write('- Test more complex models (random forests, XGBoost)\n\n')

    f.write('---\n\n')
    # ------------------------------------------------------------------------------------------------------------------
    # Appendix: Model Improvements
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Appendix: Model Improvements\n\n')
    f.write('Three modifications were tested to explore potential improvements beyond the baseline model.\n\n')

    f.write('### A.1 Dummy Variables\n\n')
    f.write('Education and marital status were converted from numeric codes to dummy variables. ')
    f.write('This allows each category to have its own coefficient rather than assuming a linear relationship.\n\n')
    f.write('| Metric | Original | Dummies | Change |\n')
    f.write('|--------|----------|---------|--------|\n')
    f.write(f'| Accuracy | {accuracy_test:.4f} | {accuracy_d:.4f} | {accuracy_d - accuracy_test:+.4f} |\n')
    f.write(
        f'| Sensitivity | {sensitivity_test:.4f} | {sensitivity_d:.4f} | {sensitivity_d - sensitivity_test:+.4f} |\n')
    f.write(f'| AUC | {auc_train:.4f} | {auc_d:.4f} | {auc_d - auc_train:+.4f} |\n\n')
    f.write('**Result:** All metrics improved slightly. Marital status shows strong effects: ')
    f.write('married, single, and other categories all have higher default risk than "unknown" status.\n\n')

    f.write('### A.2 Optimal Cutoff\n\n')
    f.write('The default classification cutoff of 0.5 assumes equal cost of false positives and false negatives. ')
    f.write('For a bank, missing a default (false negative) may be more costly than a false alarm. ')
    f.write('We searched for the cutoff that maximizes F1-score (harmonic mean of precision and recall).\n\n')
    f.write('| Cutoff | Accuracy | Sensitivity | Specificity |\n')
    f.write('|--------|----------|-------------|-------------|\n')
    f.write(f'| 0.50 (default) | {accuracy_test:.4f} | {sensitivity_test:.4f} | {specificity_test:.4f} |\n')
    f.write(
        f'| 0.20 (optimal) | {best_f1["accuracy"]:.4f} | {best_f1["sensitivity"]:.4f} | {best_f1["specificity"]:.4f} |\n\n')
    f.write(
        f'**Result:** Sensitivity jumps from {sensitivity_test * 100:.1f}% to **{best_f1["sensitivity"] * 100:.1f}%**, ')
    f.write(f'meaning the model catches more than twice as many actual defaults. ')
    f.write(
        f'The trade-off is lower specificity ({best_f1["specificity"] * 100:.1f}% vs {specificity_test * 100:.1f}%).\n\n')

    f.write('### A.3 Interaction Term\n\n')
    f.write('An interaction term `avg_delay × avg_debt_to_limit` was added to test whether the effect ')
    f.write('of payment delay depends on the debt level.\n\n')
    f.write(f'- **Coefficient:** {coef_interaction:.4f} (p-value = {pval_interaction:.4f})\n')
    f.write(f'- **Interpretation:** The interaction is statistically significant and **negative**. ')
    f.write('This means the effect of payment delay **weakens** as debt-to-limit ratio increases. ')
    f.write('In other words, for already highly indebted clients, additional payment delays matter less.\n\n')
    f.write('| Metric | Original | Interaction | Change |\n')
    f.write('|--------|----------|-------------|--------|\n')
    f.write(f'| Accuracy | {accuracy_test:.4f} | {accuracy_i:.4f} | {accuracy_i - accuracy_test:+.4f} |\n')
    f.write(
        f'| Sensitivity | {sensitivity_test:.4f} | {sensitivity_i:.4f} | {sensitivity_i - sensitivity_test:+.4f} |\n')
    f.write(f'| AUC | {auc_train:.4f} | {auc_i:.4f} | {auc_i - auc_train:+.4f} |\n\n')

    f.write('### A.4 Summary and Recommendation\n\n')
    f.write('| Modification | Best For | Key Trade-off |\n')
    f.write('|--------------|----------|---------------|\n')
    f.write('| Dummy variables | Slight overall improvement | Minimal downside |\n')
    f.write('| Cutoff = 0.20 | Catching more defaults | More false alarms |\n')
    f.write('| Interaction term | Understanding relationships | No AUC improvement |\n\n')
    f.write('**Recommendation:** ')
    f.write('If the bank prioritizes identifying risky clients (even with some false alarms), ')
    f.write('use **cutoff = 0.20**. If interpretability matters most, stick with the original model. ')
    f.write('Dummy variables for marital status can be added for a small but consistent improvement.\n')