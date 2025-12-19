# Lab 9: Implement CI/CD Pipelines for ML with GitHub Actions

## Lab overview
In this lab, you will:
- Set up a CI/CD pipeline for the Iris dataset with GitHub actions
- Train a machine learning model using scikit-learn
- Version and deploy the model using Continuous Machine Learning
- Deploy to GitHub pages from the automated workflow

**Estimated completion time**
25 minutes

129

---

### Task 1: Setting up the project repository and loading the dataset
In this task, you will create a GitHub repository and set up the basic project folder structure for this lab.

1. Open a Web browser, log in to GitHub (if necessary), and create a new public repository: e.g., iris-html-ci-cd add a description, and initialize with a README.
2. Open the Visual Studio Code, open a new terminal window and clone the repository locally, using this command:

```bash
git clone https://github.com/your-username/iris-html-ci-cd.git
```
3. Using Windows Explorer, copy all of the following files: `ci.yml`, `generate_html.py`, `requirements.txt` and `train_model.py` from the `C:\MLOps\Lab-Files` folder, into the new `iris-html-ci-cd` project folder: `C:\Users\student\iris-html-ci-cd`.
4. In VS Code, click File > Open Folder and search for the new project folder location (`C:\Users\student\iris-html-ci-cd`). The folder will show in the left window of the Visual Studio Code. Now click on the Terminal menu item and then the sub-menu item, New Terminal. This will open a terminal.
5. Create a new virtual environment for this task. Run the following command in the terminal window to create the virtual environment.

```bash
virtualenv venv
```
6. Run the following command to activate the virtual environment.

```powershell
.\venv\Scripts\activate
```
7. Now, run the following command to confirm that the virtual environment is activated.

```bash
python --version
```
8. A new virtual environment setup is completed for this project. To install the required libraries in the virtual environment, run the following command in the terminal window.

```bash
pip install scikit-learn pandas matplotlib
```
9. Create/verify the necessary folder structure, as below (create the directories and move the files, as required, in either Windows Explorer or VS Code).

```
iris-html-ci-cd/
├── Data/
└── iris.data
├── train_model.py
├── generate_html.py
├── requirements.txt
├── .github/
└── workflows/
└── ci.yml
├── index.html

# This file will be auto-generated.

├── README.md
```

10. Copy the `iris.data` file from `C:\MLOps\Data-Files` into the project directory `Data/` folder. Alternatively, execute the below command in the terminal window to automatically download the dataset in the data folder.

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P Data/
```

---

### Task 2: Training a model and generating an HTML app
The goal of this task is to process the dataset, train a model, and write a script to automatically generate the HTML file in the project folder to be used later to show the results.

1. Click on the file with the name `train_model.py` in the project folder and check/verify that the following code is present.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# Load the Iris dataset
df = pd.read_csv('Data/iris.data', header=None)

df.columns = ['sepal_length', 'sepal_width', 'petal_length',
'petal_width', 'species']

# Split the data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save results to a JSON file
results = {
"accuracy": accuracy,
"feature_importances": list(model.feature_importances_)
}
with open("results.json", "w") as f:
json.dump(results, f)

print("Model training completed. Results saved to results.json.")
```

2. Run the script by executing the following command in the terminal.

```bash
python train_model.py
```
3. Verify that `results.json` is generated and contains model accuracy and feature importances after executing the above model training script.
4. Now click on the Python file `generate_html.py` in the project folder and verify the contents, as below. This script will read the `results.json` file and generate an HTML file (`index.html`) for the app.

```python
import json
# Load results from JSON
with open("results.json", "r") as f:
results = json.load(f)
# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initialscale=1.0">
<title>Iris Model Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h1 {{ color: #4CAF50; }}
.results {{ margin-top: 20px; }}
</style>
</head>
<body>

<h1>Iris Model Results</h1>
<div class="results">
<p><strong>Accuracy:</strong> {results['accuracy']:.2f}</p>
<p><strong>Feature Importances:</strong></p>
<ul>
<li>Sepal Length:
{results['feature_importances'][0]:.2f}</li>
<li>Sepal Width:
{results['feature_importances'][1]:.2f}</li>
<li>Petal Length:
{results['feature_importances'][2]:.2f}</li>
<li>Petal Width:
{results['feature_importances'][3]:.2f}</li>
</ul>
</div>
</body>
</html>
"""
# Save HTML to a file
with open("index.html", "w") as f:
f.write(html_content)

print("HTML file generated: index.html")
```

5. Run the above Python script by executing the following command in the terminal window.

```bash
python generate_html.py
```
6. Verify that `index.html` is generated with the model results (Open the file from Windows Explorer, which will open a browser window).

---

### Task 3: Automating CI/CD with GitHub actions
This task aims to set up a CI/CD pipeline that pushes the files to the GitHub and deploy on the GitHub pages.

1. Click on the `ci.yml` file, which should be now located in the `.github/workflows` folder, and confirm that the content is as below.

```yaml
name: CI/CD for Hosting Iris HTML App

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt

      - name: Train the model
        run: |
          source venv/bin/activate
          python train_model.py

      - name: Generate HTML app
        run: |
          source venv/bin/activate
          python generate_html.py

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
```

2. Click on the `requirements.txt` file and confirm that the following dependencies are listed in it.

```
pandas
scikit-learn
matplotlib
```

3. To commit and push changes to the GitHub repository, execute the following commands in the terminal window.

```bash
git add .
git commit -m "Set up CI/CD pipeline for GitHub Pages"
git push origin main
```
4. To enable GitHub Pages, go to the GitHub repository settings, select Pages, set the source to the Deploy from a Branch, the Branch should be set to Main and / (root). Click Save.
5. You should now see a message in the GitHub pages section that the site is live. Click on the Visit site button to view your site in a browser window or access the URL e.g., https://yourusername.github.io/iris-html-ci-cd/.

---

## Lab review
1. Which command is used to push the data to GitHub?

A. git add
B. git push origin main
C. github push repo main
D. git commit -m "Push to GitHub"

**STOP**

You have successfully completed this lab.

137