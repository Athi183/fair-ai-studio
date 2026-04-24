import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for plots
plt.style.use('ggplot')

def calculate_fairness_metrics(df, sensitive_feature='gender', target='shortlisted', prediction='prediction'):
    """
    Calculate common fairness metrics.
    Assuming privileged=1, unprivileged=0
    """
    # Filter groups
    privileged = df[df[sensitive_feature] == 1]
    unprivileged = df[df[sensitive_feature] == 0]
    
    # Selection Rates
    sr_p = privileged[prediction].mean()
    sr_u = unprivileged[prediction].mean()
    
    # 1. Disparate Impact (DI)
    # Goal: Close to 1.0. < 0.8 is usually considered biased.
    di = sr_u / sr_p if sr_p > 0 else 0
    
    # 2. Statistical Parity Difference (SPD)
    # Goal: Close to 0.
    spd = sr_u - sr_p
    
    # 3. Equal Opportunity Difference (EOD)
    # True Positive Rate (TPR) difference
    tpr_p = privileged[privileged[target] == 1][prediction].mean()
    tpr_u = unprivileged[unprivileged[target] == 1][prediction].mean()
    eod = tpr_u - tpr_p
    
    return {
        "selection_rate_privileged": float(sr_p),
        "selection_rate_unprivileged": float(sr_u),
        "disparate_impact": float(di),
        "statistical_parity_difference": float(spd),
        "equal_opportunity_difference": float(eod)
    }

def run_shap_analysis(model, X):
    """
    Run SHAP analysis and return summary plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For RandomForest in sklearn, shap_values is a list [contrib_class_0, contrib_class_1]
    # We want class 1 (shortlisted)
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1]
    else:
        # Some versions or simpler models return a single array for binary
        shap_values_class_1 = shap_values
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class_1, X, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    return shap_values_class_1

def main():
    print("Loading data and model...")
    df = pd.read_csv('cleaned_data.csv')
    model = joblib.load('biased_model.pkl')
    
    # Features used for the model
    # Based on check_model.py, there are 5 features
    feature_cols = ['gender', 'age', 'education_level', 'experience_years', 'screening_score']
    X = df[feature_cols]
    
    # 1. Calculate Fairness Metrics
    print("Calculating fairness metrics...")
    metrics = calculate_fairness_metrics(df)
    
    # 2. Run SHAP
    print("Running SHAP analysis...")
    run_shap_analysis(model, X)
    
    # 3. Save Results
    print("Saving results...")
    with open('audit_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Audit Summary ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nSHAP summary plot saved as 'shap_summary.png'")
    print("Metrics saved as 'audit_results.json'")

if __name__ == "__main__":
    main()
