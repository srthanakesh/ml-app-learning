import streamlit as st
import pandas as pd
import numpy as np
st.title('🖥️ Machine Learning App')

st.info('This is app builds a machine learning model!')

df = pd.read_csv('/penguins_cleaned.csv')
df
