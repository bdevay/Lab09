import json
# Load results from JSON
with open("results.json", "r")  as f:
    results = json.load(f)

# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Model Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #4CAF50; }}
        .results {{margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>Iris Model Results</h1>
    <div class="results">
        <p><strong>Accuracy:</strong> {results['accuracy']:.2f}</p>
        <p><strong>Feature Importances:</strong></p>
        <ul>
            <li>Sepal Length: {results['feature_importances'][0]:.2f}</li>
            <li>Sepal Width: {results['feature_importances'][1]:.2f}</li>
            <li>Petal Length: {results['feature_importances'][2]:.2f}</li>
            <li>Petal Width: {results['feature_importances'][3]:.2f}</li>
        </ul>
</body>
</html>
"""

# Save HTML to a file
with open("index.html", "w") as f:
    f.write(html_content)

print("HTML file generated: index.html")