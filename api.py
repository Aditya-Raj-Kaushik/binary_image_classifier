from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
from predict import predict_fruit

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fruit Prediction API!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Predict the fruit based on the image
    prediction = predict_fruit(contents)
    
    return {"prediction": prediction}
