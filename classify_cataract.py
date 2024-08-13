import pickle
from features_extraction_cataract import extract_features

with open('./models/cataract_classifier.pkl', 'rb') as cataract_classifier:
    cataract_clf = pickle.load(cataract_classifier)

disease_types = ["Cortical Cataract", "Nuclear Cataract", "Subcapsular Cataract"]

def predict_cataract(image):
    features, urls = extract_features(image)
    prediction = cataract_clf.predict([features])
    return disease_types[int(prediction[0])], features, urls