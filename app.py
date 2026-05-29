import streamlit as st
import joblib

# Loading the saved artifacts [6, 7]
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('LR_model.jb')

st.title("Fake News Detector") 
st.write("Enter a news article below to check whether it is fake or real") 

user_input = st.text_area("Enter News Content") 

if st.button("Check"): 
    if user_input:
        # Transforming and predicting [7]
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        
        # Displaying results: 1 is Real, 0 is Fake [7]
        if prediction == 1:
            st.success("The news is real")
        else:
            st.error("The news is fake")
    else:
        st.warning("Please enter any text to analyze") 