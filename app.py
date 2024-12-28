from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Feature names and descriptions
FEATURES = [
    ('radius_mean', 'Mean of distances from center to points on the perimeter'),
    ('texture_mean', 'Standard deviation of gray-scale values'),
    ('perimeter_mean', 'Mean size of the core tumor'),
    ('area_mean', 'Mean area of the tumor'),
    ('smoothness_mean', 'Mean of local variation in radius lengths'),
    ('compactness_mean', 'Mean of perimeter^2 / area - 1.0'),
    ('concavity_mean', 'Mean of severity of concave portions of the contour'),
    ('concave points_mean', 'Mean for number of concave portions of the contour'),
    ('symmetry_mean', 'Mean of symmetry of the tumor'),
    ('fractal_dimension_mean', 'Mean for "coastline approximation" - 1'),
    ('radius_se', 'Standard error for the mean of distances from center to points'),
    ('texture_se', 'Standard error for texture'),
    ('perimeter_se', 'Standard error for perimeter'),
    ('area_se', 'Standard error for area'),
    ('smoothness_se', 'Standard error for smoothness'),
    ('compactness_se', 'Standard error for compactness'),
    ('concavity_se', 'Standard error for concavity'),
    ('concave points_se', 'Standard error for concave points'),
    ('symmetry_se', 'Standard error for symmetry'),
    ('fractal_dimension_se', 'Standard error for fractal dimension'),
    ('radius_worst', 'Worst or largest mean value for radius'),
    ('texture_worst', 'Worst or largest mean value for texture'),
    ('perimeter_worst', 'Worst or largest mean value for perimeter'),
    ('area_worst', 'Worst or largest mean value for area'),
    ('smoothness_worst', 'Worst or largest mean value for smoothness'),
    ('compactness_worst', 'Worst or largest mean value for compactness'),
    ('concavity_worst', 'Worst or largest mean value for concavity'),
    ('concave points_worst', 'Worst or largest mean value for concave points'),
    ('symmetry_worst', 'Worst or largest mean value for symmetry'),
    ('fractal_dimension_worst', 'Worst or largest mean value for fractal dimension')
]

# Sample data for testing (a benign case)
SAMPLE_DATA = [
    17.99, 10.38, 122.8, 1001.0, 0.11840, 0.27760, 0.30010, 0.14710, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
    0.11840, 0.27760, 0.30010, 0.14710, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4
]

# Load all models
models = {
    'SVC (Linear)': joblib.load('svm_model.pkl'),
    'KNN': joblib.load('knn_model (1).pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl')
}

# Load the scaler
scaler = joblib.load('robust_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', 
                         features=FEATURES, 
                         models=models.keys(),
                         sample_data=SAMPLE_DATA)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get selected model
        selected_model = request.form.get('model')
        if selected_model not in models:
            return render_template('index.html', 
                                error='Invalid model selection',
                                models=models.keys(),
                                features=FEATURES)
        
        # Get feature values
        feature_values = []
        for i in range(len(FEATURES)):
            value = request.form.get(f'feature_{i}')
            if not value:  # Check if value is empty or None
                return render_template('index.html',
                                    error=f'Missing value for {FEATURES[i][0]}',
                                    models=models.keys(),
                                    features=FEATURES)
            try:
                feature_values.append(float(value))
            except ValueError:
                return render_template('index.html',
                                    error=f'Invalid value for {FEATURES[i][0]}. Please enter a valid number.',
                                    models=models.keys(),
                                    features=FEATURES)
        
        # Validate number of features
        if len(feature_values) != len(FEATURES):
            return render_template('index.html',
                                error='Incorrect number of features provided.',
                                models=models.keys(),
                                features=FEATURES)
        
        # Prepare and scale features
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Make classification
        model = models[selected_model]
        classification = model.predict(X_scaled)[0]
        result = 'Malignant' if classification == 1 else 'Benign'
        
        return render_template('index.html',
                             models=models.keys(),
                             features=FEATURES,
                             values=feature_values,
                             selected_model=selected_model,
                             classification=result)
    
    except ValueError as e:
        return render_template('index.html',
                             error=f'Invalid input: {str(e)}. Please enter valid numeric values.',
                             models=models.keys(),
                             features=FEATURES)
    except Exception as e:
        return render_template('index.html',
                             error=f'An error occurred: {str(e)}',
                             models=models.keys(),
                             features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True)
