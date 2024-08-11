import pickle
from features_extraction_cataract import extract_features

with open('./models/cataract_classifier.pkl', 'rb') as cataract_classifier:
    cataract_clf = pickle.load(cataract_classifier)

disease_types = ["Cataract 1", "Cataract 2", "Cataract 3"]

def predict_cataract(image):
    features = extract_features(image)
    prediction = cataract_clf.predict([features])
    return disease_types[prediction[0]], features