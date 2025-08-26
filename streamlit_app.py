import streamlit as st
import pandas as pd
import numpy as np
st.title('🖥️ Machine Learning App')

st.info('This is app builds a machine learning model!')
with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('penguins_cleaned.csv')
  df
  st.write("**X**")
  X = df.drop('species',axis=1)
  X

  st.write('**y**')
  y = df.species
  y

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe','Dream','Torgersen'))  
  bill_length_mm = st.slider('Bill Length (mm)',32.1,59.6,43.9)
  bill_depth_mm = st.slider('Bill Depth (mm)',13.1,21.5,17.2)
  flipper_length_mm = st.slider('Flipper Length (mm)',172.0,231.0,201.0)
  body_mass_g = st.slider('Body Mass (g)',2700.00,6300.0,4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))

  data = {
    'island',island,
    'bill_length_mm', bill_length_mm,
    'bill_depth_mm',bill_depth_mm,
    'flipper_length_mm',flipper_length_mm,
    'body_mass_g',body_mass_g,
    'gender ',gender  
  }
  input_df = pd.DataFrame(data, index[0])
  input_df

  
                        
