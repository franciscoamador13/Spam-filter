from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import email
from email_transformers import EmailToWordCounterTransformer, WordCounterToVectorTransformer

app = FastAPI()

model = joblib.load("models/spam_model.pkl")
pipeline = joblib.load("models/pipeline.pkl")

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: EmailRequest):
    msg = email.message_from_string(request.text)
    transformed = pipeline.transform([msg])
    prediction = model.predict(transformed)[0]
    return {"spam": bool(prediction)}