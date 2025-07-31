import streamlit as st
import numpy as np
import pickle as pkl

# Load data with caching
@st.cache_data
def load_model():
    with open("iris_dataset.pkl", 'rb') as f:
        model = pkl.load(f)
    return model

model = load_model()

# Mapping class labels to names
iris_class_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Set up the app title
st.title('Iris Flower Prediction')

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5)

# Prepare input array for prediction
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict button
if st.button("Predict Iris Type"):
    prediction = model.predict(input_features)
    predicted_class = prediction[0]
    # Map the prediction to the flower name
    flower_name = iris_class_mapping[predicted_class]
    st.write(f"Predicted Iris species: {flower_name}")
