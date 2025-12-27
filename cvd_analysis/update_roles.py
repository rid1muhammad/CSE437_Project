import json

notebook_path = r'f:\venv\cvd_analysis\main_ensemble_2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Target the first cell
cell = nb['cells'][0]

# Check if we already added it (idempotency)
source = cell['source']
connected_source = "".join(source)

if "Md.Tanvir Haque Shitab" in connected_source:
    print("Roles already present.")
else:
    # Append the new section
    # Ensure the last line has a newline
    if source and not source[-1].endswith('\n'):
        source[-1] = source[-1] + '\n'
    
    new_section = [
        "\n",
        "## Team Contributions\n",
        "This project was collaboratively developed by a 3-member team. Each member was assigned a clearly defined module of the system to ensure equal contribution and accountability. The responsibilities were divided as follows:\n",
        "\n",
        "**Md.Tanvir Haque Shitab – Dataset Handling**\n",
        "*   Downloaded the Cardiovascular dataset from Kaggle\n",
        "*   Performed dataset exploration and checking for duplicates\n",
        "*   Implemented data cleaning (filtering invalid blood pressure, height, and weight values)\n",
        "*   Prepared the dataset for modeling (age conversion, scaling)\n",
        "\n",
        "**Muhammad Ridwan – Model Training**\n",
        "*   Defined feature vectors and target variables\n",
        "*   Implemented a diverse set of classifiers: Logistic Regression, KNN, SVM, Decision Tree, Naive Bayes, Random Forest, Gradient Boosting, and AdaBoost\n",
        "*   Developed the Ensemble Voting Classifier to aggregate predictions\n",
        "\n",
        "**Tasin Ahsan Rhidy – Evaluation**\n",
        "*   Performed train-test split for validation\n",
        "*   Analyzed the effect of smoking on CVD prevalence (Odds Ratio analysis)\n",
        "*   Evaluated models using Accuracy scores, Classification Reports, and Confusion Matrices\n",
        "*   Generated visualization plots for feature importance and model comparison"
    ]
    
    cell['source'].extend(new_section)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook updated successfully.")
