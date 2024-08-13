from classify_glaucoma import predict_glaucoma
from classify_diabetes_retinopathy import predict_diabetes_retinopathy
from classify_cataract import predict_cataract
from classify_retinal_detachment import predict_retinal_detachment
from extract_other_fundus import extract_common_features

def predict_disease(img):
    predict = None
    if False:
        predict = predict_glaucoma
    elif False:
        predict = predict_diabetes_retinopathy
    elif False:
        predict = predict_cataract
    elif True:
        predict = predict_retinal_detachment
    else:
        predict = extract_common_features
    return predict(img)