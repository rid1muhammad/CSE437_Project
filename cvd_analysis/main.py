import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    df['age_years'] = (df['age'] / 365.25).round(1)
    
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"Removing {dupes} duplicate rows...")
        df.drop_duplicates(inplace=True)

    mask = (df['ap_hi'] >= 60) & (df['ap_hi'] <= 240) & \
           (df['ap_lo'] >= 30) & (df['ap_lo'] <= 160) & \
           (df['ap_hi'] > df['ap_lo'])
    
    df_clean = df[mask].copy()
    print(f"Rows after blood pressure cleaning: {df_clean.shape[0]} (removed {df.shape[0] - df_clean.shape[0]})")
    
    df_clean = df_clean[(df_clean['height'] > 100) & (df_clean['weight'] > 30)]
    
    print(f"Final shape after preprocessing: {df_clean.shape}")
    return df_clean

def analyze_smoking_effect(df):
    print("\n--- Analysis: Effect of Smoking on CVD ---")
    
    prevalence = df.groupby('smoke')['cardio'].mean()
    print("\nCVD Prevalence:")
    print(f"Non-Smokers (0): {prevalence[0]*100:.2f}%")
    print(f"Smokers (1):     {prevalence[1]*100:.2f}%")
    
    ct = pd.crosstab(df['smoke'], df['cardio'])
    odds_smoker = ct.iloc[1, 1] / ct.iloc[1, 0]
    odds_nonsmoker = ct.iloc[0, 1] / ct.iloc[0, 0]
    or_val = odds_smoker / odds_nonsmoker
    
    print("\nUnadjusted Odds Ratio (Smoker vs Non-Smoker):")
    print(f"OR: {or_val:.4f}")
    if or_val > 1:
        print("Interpretation: Smokers have higher odds of CVD compared to non-smokers (in this unadjusted view).")
    else:
        print("Interpretation: Smokers have lower/equal odds of CVD compared to non-smokers (unexpected, likely due to age confounding).")

def train_and_evaluate(df):
    print("\n--- Model Training & Evaluation ---")
    
    features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'
    
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM": LinearSVC(random_state=42, dual=False),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        
        if name == "Logistic Regression":
            print("Logistic Regression Coefficients (Feature Importance):")
            coeffs = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_[0]
            }).sort_values(by='Coefficient', ascending=False)
            print(coeffs)
            
            smoke_coef = coeffs[coeffs['Feature'] == 'smoke']['Coefficient'].values[0]
            print(f"\nLogistic Regression Coefficient for 'smoke': {smoke_coef:.4f}")
            print(f"Adjusted Odds Ratio for 'smoke': {np.exp(smoke_coef):.4f}")

            # Save Feature Importance Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Coefficient', y='Feature', data=coeffs)
            plt.title('Feature Importance (Logistic Regression)')
            plt.tight_layout()
            plt.savefig('f:/venv/cvd_analysis/plots/feature_importance.png')
            plt.close()
        
        # Save Confusion Matrix for Random Forest (or best model)
        if name == "Random Forest":
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(f'f:/venv/cvd_analysis/plots/confusion_matrix_{name.replace(" ", "_")}.png')
            plt.close()

def optimize_visuals(df):
    """
    Generate general plots.
    """
    # Smoking vs Cardio
    plt.figure(figsize=(6, 5))
    sns.countplot(x='smoke', hue='cardio', data=df)
    plt.title('CVD Cases by Smoking Status')
    plt.xlabel('Smoking Status (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.legend(title='CVD', labels=['No', 'Yes'])
    plt.savefig('f:/venv/cvd_analysis/plots/smoking_impact.png')
    plt.close()

if __name__ == "__main__":
    data_path = "f:/venv/cvd_analysis/cardio_train.csv"
    output_path = "f:/venv/cvd_analysis/results.txt"
    
    with open(output_path, "w") as f:
        sys.stdout = f
        try:
            df = load_and_preprocess(data_path)
            optimize_visuals(df) # Generate general plots
            analyze_smoking_effect(df)
            train_and_evaluate(df)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
            
    with open(output_path, "r") as f:
        print(f.read())
