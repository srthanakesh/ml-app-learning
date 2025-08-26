import streamlit as st

st.title('🖥️ Machine Learning App')

st.info('This is app builds a machine learning model!')

df = pd.read_csv('/penguins_cleaned.csv')
df
