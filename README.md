# ⚖️ FairAI Studio

**FairAI Studio** is a comprehensive tool designed to detect, audit, and mitigate bias in AI-driven recruitment processes. Using advanced Fairness Metrics and Explainable AI (XAI), this platform ensures that machine learning models make decisions based on merit, not demographics.

Built for the **Google GDG Challenge**, this project demonstrates a collaborative approach to Responsible AI.

---

## 🚀 Key Features

- **🔍 Automated Bias Audit:** Calculate Disparate Impact, Statistical Parity, and Equal Opportunity metrics.
- **🧠 Explainable AI (XAI):** Leverage SHAP visualizations to understand feature influence (e.g., Experience vs. Gender).
- **🛠️ Bias Mitigation:** Active resampling and re-weighting techniques to balance skewed datasets.
- **📊 Interactive Dashboard:** A modern, dark-mode platform built with FastAPI and Premium UI components for real-time monitoring.

---

## 🛠️ Tech Stack

- **Backend:** Python (FastAPI)
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** Modern Web UI (Member 4 Implementation)

---

## 👥 Collaborative Team Structure

The project is divided into four expert modules:

1. **Member 1 (Data & Model):** Baseline model training and data preprocessing.
2. **Member 2 (Forensic Auditor):** Bias metrics and SHAP explainability analysis.
3. **Member 3 (Fairness Engineer):** Implementing mitigation strategies and fair model retraining.
4. **Member 4 (Platform Architect):** Web integration and interactive dashboard development.

---

## 📈 Initial Audit Results
*Current baseline model performance:*
- **Disparate Impact:** 1.10
- **Statistical Parity Difference:** 0.03
- **Primary Drivers:** Professional Experience and Screening Scores.

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/FathimaMehrinVS/fair-ai-studio.git

# Install dependencies
pip install -r requirements.txt

# Run the Audit (Member 2 Module)
python bias_auditor.py
```

---

*This project is part of a dedicated effort to promote Fairness and Transparency in AI.*
