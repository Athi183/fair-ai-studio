import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

# Features
gender = np.random.choice([0, 1], size=n, p=[0.5, 0.5]) # 0=Female, 1=Male
age = np.random.randint(22, 60, size=n)
education_level = np.random.randint(0, 5, size=n) # 0 to 4
experience_years = np.clip(age - 22 + np.random.randint(-2, 3, size=n), 0, 30)
screening_score = np.random.normal(65, 15, size=n)
screening_score = np.clip(screening_score, 20, 100)

logit = (education_level * 1.5) + (experience_years * 0.3) + (screening_score * 0.1) - 18
biased_logit = logit + (gender * 4) 
prob = 1 / (1 + np.exp(-biased_logit))
shortlisted = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'gender': gender,
    'age': age,
    'education_level': education_level,
    'experience_years': experience_years,
    'screening_score': screening_score,
    'shortlisted': shortlisted
})

df.to_csv("cleaned_data.csv", index=False)
print("Regenerated highly tuned realistic dataset with embedded bias.")
