import pickle
from features_extraction_diabetes_retinopathy import extract_features

with open('./models/diabetes_retinopathy_classifier.pkl', 'rb') as diabetes_retinopathy_classifier:
    diabetes_retinopathy_clf = pickle.load(diabetes_retinopathy_classifier)

disease_types = ["Primary Open Angle Glaucoma", "Normal Tension Glaucoma", "Pigmentary Glaucoma"]

def predict_diabetes_retinopathy(image):
    features = extract_features(image)
    prediction = diabetes_retinopathy_clf.predict([features])
    return disease_types[prediction[0]], features