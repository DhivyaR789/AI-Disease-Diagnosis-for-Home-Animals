import pandas as pd  
import numpy as np 
import pickle as pkl  
import streamlit as st

pickle_mod = open('savemodel.sav', 'rb')  
classifier = pkl.load(pickle_mod)  

def match(s1,s2,s3,s4,s5,s6,s7):
    symp1 = {0:"Aggression",1:"Apathy",3:"Lethergic",2:"Lameness",4:"Stiffness"}
    symp2 = {0:"Pale mucus membrane	",1:"Retching",2:"Swalow difficult",3:"Vomiting"}
    symp3 = {0:"Anorexia", 1:"Diarrhoea", 2:"Honking cough",3:"Paralysis"}
    symp4 = {0:"Dark color urine",1:"Dehydration",2:"Nasal Discharge",3:"Nausea",4:"Nose bleeding",5:"Running Nose",6:"Staggering",7:"Yellowish urine"}
    symp5 = {0:"Anarmic",1:"Inappetance",2:"Pustules",3:"Seizures",4:"Swelling",5:"Weakness",6:"Weight loss"}
    symp6 = {1:"Anaemia",0:"Abdominal pain",2:"Convulsion",3:"Depression",4:"Drooling",5:"Enlarged lymphnodes",6:"Pale mucus membrane",7:"Sneeze"}
    symp7 = {0:"fever",1:"NaN",2:"No fever",3:"Poor general condition"}
    
    def encode(list1,dict1):
        for i in range(len(dict1)):
            if(list1 == dict1[i]):
                count = i
                break
        else:
            count = 0
        if(count == 0):
            return 1
        else: 
            return count 
        
    s1 = encode(s1,symp1)
    s2 = encode(s2,symp2)
    s3 = encode(s3,symp3)
    s4 = encode(s4,symp4)
    s5 = encode(s5,symp5)
    s6 = encode(s6,symp6)
    s7 = encode(s7,symp7)
    
    dis_enc = []
    dis_enc.extend([s1,s2,s3,s4,s5,s6,s7])
    
    disease_df = pd.DataFrame(dis_enc).T
    disease_df.columns = ['Symp1','Symp2','Symp3','Symp4','Symp5','Symp6','Symp7']
    return disease_df
    


 
def main():
    st.title('AI Disease Diagnosis')
    html = """
    <div style = "background-colour: #FFFF00; padding: 16px">  
    <h1 style = "color: #000000; text-align: centre; "> Animal Disease Diagnosis ML App   
     </h1>  
    </div>  """
    
    st.markdown(html,unsafe_allow_html = True)
    
    symp1 = st.text_input ("Symptom1 ")  
    symp2 = st.text_input ("Symptom2 ")  
    symp3 = st.text_input ("Symptom3 ")  
    symp4 = st.text_input ("Symptom4 ")  
    symp5 = st.text_input ("Symptom5 ")  
    symp6 = st.text_input ("Symptom6 ")  
    symp7 = st.text_input ("Symptom7 ")  

    result = " "  
    
    if st.button('Diagonse'):
        result = match(symp1,symp2,symp3,symp4,symp5,symp6,symp7)
        result = classifier.predict(result)
        st.success('It may be due to {}'.format(result))
        
        
if __name__ == '__main__':
    main()