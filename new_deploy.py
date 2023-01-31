import streamlit as st
import numpy as np
import pickle
from keybert import KeyBERT
import pandas as pd

pickle_mod = open('savemodel.sav','rb')
classify = pickle.load(pickle_mod)

def preprocess(text):
    model = KeyBERT()
    words = model.extract_keywords(text)
    
    list1 = []
    for i in words:
        list1.append(i)
        
    split = []
    for i in range(len(list1)):
        split_arg = list1[i][0].split()
        split.append(split_arg)
    split_list = [j for sub in split for j in sub]
    
    disease = []
    X = [['Lethergic','Vomiting','Diarrhorea','Nasal Discharge','Pustules','Convulsion','Fever'],
         ['Lethergic','Retching','Honking cough','Running Nose','Inappetance','Sneeze','Fever'],
         ['Aggression','Swalow difficult','Paralysis','Staggering','Seizures','Drooling','No fever'],
         ['Apathy','Pale mucus membrane','Anorexia','Dark color urine','Anarmic','Pale mucus membrane','Poor general condition'],
         ['Lethergic','Vomiting','Diarrhorea','Nose bleeding','Weight loss','Enlarged lymphnodes','Fever'],
         ['Lameness', 'Vomiting', 'Diarrhorea', 'Nausea', 'Swelling', 'Anaemia', 'No fever'],
         ['Stiffness','Vomiting','Diarrhorea','Yellowish urine','Weakness','Depression','Fever'],
         ['Lethergic','Vomiting','Diarrhorea','Dehydration','Weakness','Abdominal pain','No fever']]
    from itertools import chain
    symptoms = list(chain.from_iterable(X))

    unique_sym = list(set(symptoms))

    for i in range(len(split_list)):
        spc = split_list[i].capitalize()
        if spc in unique_sym:
            disease.append(spc)
    if len(disease) != 7:
        for i in range(7 - len(disease)):
            disease.append('fever')
            
            
    symp1 = {0:"Aggression",1:"Apathy",3:"Lethergic",2:"Lameness",4:"Stiffness"}
    symp2 = {0:"Pale mucus membrane	",1:"Retching",2:"Swalow difficult",3:"Vomiting"}
    symp3 = {0:"Anorexia", 1:"Diarrhoea", 2:"Honking cough",3:"Paralysis"}
    symp4 = {0:"Dark color urine",1:"Dehydration",2:"Nasal Discharge",3:"Nausea",4:"Nose bleeding",5:"Running Nose",6:"Staggering",7:"Yellowish urine"}
    symp5 = {0:"Anarmic",1:"Inappetance",2:"Pustules",3:"Seizures",4:"Swelling",5:"Weakness",6:"Weight loss"}
    symp6 = {1:"Anaemia",0:"Abdominal pain",2:"Convulsion",3:"Depression",4:"Drooling",5:"Enlarged lymphnodes",6:"Pale mucus membrane",7:"Sneeze"}
    symp7 = {0:"fever",2:"No fever",3:"Poor general condition"}
    
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
        
    s1 = encode(disease[0],symp1)
    s2 = encode(disease[1],symp2)
    s3 = encode(disease[2],symp3)
    s4 = encode(disease[3],symp4)
    s5 = encode(disease[4],symp5)
    s6 = encode(disease[5],symp6)
    s7 = encode(disease[6],symp7)
    
    dis_enc = []
    dis_enc.extend([s1,s2,s3,s4,s5,s6,s7])
    
    disease_df = pd.DataFrame(dis_enc).T
    disease_df.columns = ['Symp1','Symp2','Symp3','Symp4','Symp5','Symp6','Symp7']
    return disease_df

def diagnose(ip):
    process = preprocess(ip)
    disease = classify.predict(process)
    return disease

def main():
    st.title("AI Disease Diagnoiser")
    html = """<div style = "background-colour: #FFFF00; padding: 16px">  
    <h1 style = "color: #000000; text-align: centre; "> Streamlit Iris Flower Classifier ML App   
     </h1>  
    </div>  """
    
    inp = st.text_input("Symptoms")
    if st.button("Diagnose"):
        result = diagnose(inp)
        st.success('It may be due to {}'.format(result))

if __name__=='__main__':
    main()
    