from multimodal_predictor import predict_multimodal

text = "Government announces new education policy for universities."
image_path = "data/images/16.jpg"  # any image

result = predict_multimodal(text, image_path)
print(result)
