import streamlit as st 
import pandas as pd 
import pickle 

#title
st.title("Penguin Prediction Application")

#description of app
st.markdown(
    '''
    This app predicts the **Palmer Penguin** species!
    
    Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.

    ***
'''
)

#sidebar/ set input features
st.sidebar.header("User Input Features")

#file line
st.sidebar.markdown("[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)")

#link to upload file 
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

model = pickle.load(open("penguins_clf.pkl",'rb'))
st.subheader("User Input Features")
if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)


#print predictions
predictions = model.predict(input_df)
predictions_proba = model.predict_proba(input_df)

st.subheader("Predictions")
st.write(predictions)

st.subheader("predictions proba")
st.write(predictions_proba)