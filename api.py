from fastapi import FastAPI, File, UploadFile, HTTPException
from predict import predict_fruit

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fruit Prediction API!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    contents = await file.read()

    prediction, confidence = predict_fruit(contents)

    return {
        "prediction": str(prediction),
        "confidence": round(float(confidence), 4)  
    }
