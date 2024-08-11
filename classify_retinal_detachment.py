import pickle
from features_extraction_retinal_detachment import extract_features

with open('./models/retinal_detachment_classifier.pkl', 'rb') as retinal_detachment_classifier:
    retinal_detachment_clf = pickle.load(retinal_detachment_classifier)

disease_types = ["Primary Open Angle Glaucoma", "Normal Tension Glaucoma", "Pigmentary Glaucoma"]

def predict_retinal_detachment(image):
    features = extract_features(image)
    prediction = retinal_detachment_clf.predict([features])
    return disease_types[prediction[0]], features