import torch
model_path = r"C:\Users\86134\Desktop\SLOWFAST_8x8_R50_DETECTION.pyth"
model = model.load(model_path)
whiten = model.Linear(2048, 7, bias=True)

