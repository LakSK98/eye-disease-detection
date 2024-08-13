import pickle
from features_extraction_glaucoma import extract_features

with open('./models/glaucoma_classifier.pkl', 'rb') as glaucoma_classifier:
    glaucoma_clf = pickle.load(glaucoma_classifier)

disease_types = ["Primary Open Angle Glaucoma", "Normal Tension Glaucoma", "Pigmentary Glaucoma"]

def predict_glaucoma(image):
    features, processed_urls = extract_features(image)
    prediction = glaucoma_clf.predict([features])
    return "Glaucoma", disease_types[int(prediction[0])], features, processed_urls