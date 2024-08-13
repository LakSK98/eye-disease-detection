import joblib
from features_extraction_retinal_detachment import extract_features

retinal_detachment_clf = joblib.load('./models/retinal_detachment_classifier.pkl')

disease_types = ["RRD", "ERD", "TRD"]

def predict_retinal_detachment(image):
    features, urls = extract_features(image)
    prediction = retinal_detachment_clf.predict([features])
    return "Retinal Detachment", disease_types[int(prediction[0])], features, urls