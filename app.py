import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load fine-tuned model & tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

# Label mapping for BBC News Dataset
label_map = {
    0: "Business",
    1: "Entertainment",
    2: "Politics",
    3: "Sports",
    4: "Technology"
}

st.title("📰 News Article Classification App")

user_input = st.text_area("Enter a news article or headline:")

if st.button("Predict"):
    if user_input.strip():
        # Tokenize
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Map prediction to label
        predicted_label = label_map.get(predicted_class, "Unknown")

        st.success(f"📌 Predicted Category: **{predicted_label}**")
    else:
        st.warning("Please enter some text.")
