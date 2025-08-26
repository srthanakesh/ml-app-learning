import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    'island':island,
    'bill_length_mm':bill_length_mm,
    'bill_depth_mm':bill_depth_mm,
    'flipper_length_mm':flipper_length_mm,
    'body_mass_g':body_mass_g,
    'gender ':gender  
  }
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X],axis=0)

with st.expander('Input features'):
  st.write('**Input features**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins

# Data preparation 
# encode X
df_penguins = pd.get_dummies(input_penguins, columns=['island','sex'])
X = df_penguins[1:]
input_row=df_penguins[:1]

# encode Y
target_mapper ={
  'Adelie' :0,
  'Chinstrap':1,
  'Gentoo':2,  
}
def target_encode(val):
  return target_mapper[val]

y_new= y.apply(target_encode)
y
y_new

with st.expander('Data preparation'):
  st.write("**Encoded X**")
  input_row
  st.write ('**Encoded y**')
  y_new

# Model Training
clf = RandomForestClassifier()
clf.fit(X,y_new)

## Apply model 
pred = clf.predict(input_row)
pred_probab = clf.predict_proba(input_row)
df_pred = pd.DataFrame(pred_probab)
df_pred.rename(columns={0:'Adelie',1:'Chinstrap',2:'Gentoo'})


# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_pred,column_config={
  'Adelie':st.column_config.ProgressColumn(
    'Adelie',
    format='%f',
    width='medium',
    min_value=0,
    max_balue=1
  ),'Chinstrap':st.column_config.ProgressColumn(
    'Chinstrap',
    format='%f',
    width='medium',
    min_value=0,
    max_balue=1
  ),
  'Gentoo':st.column_config.ProgressColumn(
    'Gentoo',
    format='%f',
    width='medium',
    min_value=0,
    max_balue=1
  )
})
df_pred
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.success(str(penguins_species[pred][0]))





  

  
                        
