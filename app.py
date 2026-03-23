from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import email
from email_transformers import EmailToWordCounterTransformer, WordCounterToVectorTransformer

app = FastAPI()

# The loading of the trained model and preprocessing pipeline happens once when the server starts, not on every request
model = joblib.load("models/spam_model.pkl")
pipeline = joblib.load("models/pipeline.pkl")


# Define the expected structure of the incoming JSON request
# FastAPI will automatically validate that "text" is a string
class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: EmailRequest):
    # Parse the raw text into a Python email object
    # The pipeline expects this format since it was trained on email objects
    msg = email.message_from_string(request.text)
    
    # Run the email through the preprocessing pipeline:
    # 1. Extract plain text from the email
    # 2. Count word frequencies
    # 3. Convert to a numerical vector
    transformed = pipeline.transform([msg])
    
    # Predict whether the email is spam (1) or not (0)
    prediction = model.predict(transformed)[0]
    
    # Return the result as a boolean
    return {"spam": bool(prediction)}