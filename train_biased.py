import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("cleaned_data.csv")
X = df[["gender", "age", "education_level", "experience_years", "screening_score"]]
y = df["shortlisted"]

# Train the biased model
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X, y)

df["prediction"] = model.predict(X)
df.to_csv("cleaned_data.csv", index=False)

joblib.dump(model, "biased_model.pkl")
print("Biased model trained and saved.")
