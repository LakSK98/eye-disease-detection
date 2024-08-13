from features_extraction_diabetes_retinopathy import extract_features
import joblib

# Load the classifier
diabetes_retinopathy_clf = joblib.load('./models/diabetes_retinopathy_classifier.pkl')

disease_types = ["None", "Mild DR", "Moderate DR", "Severe DR", "PDR"]

def predict_diabetes_retinopathy(image):
    features, urls = extract_features(image)
    prediction = diabetes_retinopathy_clf.predict([features])
    return "Diabetes Retinopathy", disease_types[int(prediction[0])], features, urls