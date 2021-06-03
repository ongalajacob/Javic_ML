# get streamlit 

#streamlitrun app.py   IN CONSOLE

from typing import Any
import streamlit as st 
import pandas as pd 
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from bokeh.plotting import figure

import os
import seaborn as sns
import cufflinks as cf
import warnings
import cufflinks as cf
import plotly.express as px 
import plotly.graph_objects as go
import requests
import io  

from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import warnings


########################### Display text ###########################################

student = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_student.csv'
class_session = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_class_session.csv'
classregister = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_classregister.csv'
invoice = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_invoice.csv'
employees = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_employees.csv'
fees = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_fees.csv'
exams = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_exams.csv'
#model ='https://github.com/ongalajacob/Javic/blob/main/API/Javic_prediction_model.pkl'

def main():
    stud_df = pd.read_csv(student)
    session_df = pd.read_csv(class_session,dtype={'Class_grd' : 'category'})
    classregister_df = pd.read_csv(classregister)
    invoice_df = pd.read_csv(invoice)
    employees_df = pd.read_csv(employees)
    fee_df = pd.read_csv(fees)
    exams_df = pd.read_csv(exams)

    Register = pd.merge(left=classregister_df, right=session_df, how='left', left_on='Session', right_on='id')
    Register.drop(["id_y","Session"], axis=1, inplace=True)
    Register.rename(columns = {'id_x':'id', }, inplace = True)

    Register = pd.merge(left=Register, right=stud_df, how='left', left_on='student', right_on='ID')
    Register.drop(['ID','Mother', 'Father', 'Guadian',
        'Class_Admitted', 'DOA', 'address',
            'DateExit',"student",'Startdate','Enddate','ClassTeacher'], axis=1, inplace=True)
    Register.rename(columns = {'sex_x':'sex_teacher','sex_y':'sex_stud' ,'name':'name_stud' ,'Class_grd':'grade'  }, inplace = True)

    Register=pd.merge(left=Register, right=invoice_df, how='left', left_on='id', right_on='ClassRegisterID')
    Register.drop(["id_y"], axis=1, inplace=True)
    Register.rename(columns = {'id_x':'id', }, inplace = True)
    Register_df = Register.rename(columns=str.lower)
    Register_df['tot_invoice']=Register_df['uniform']+Register_df['uniform_no']+(Register_df['transport']*Register_df['transport_months'])+ \
        (Register_df['lunch']*Register_df['lunch_months'])+Register_df['otherlevyindv']+Register_df['tutionfee']+Register_df['examfee']+ \
            Register_df['booklevy'] +Register_df['activityfee']+Register_df['otherlevies']
    Register_df= Register_df[['id', 'year', 'term', 'grade', 
        'name_stud', 'adm', 'dob', 'sex', 'phone1', 'phone2', 'phone3', 'enrolstatus', 'bal_bf', 'tot_invoice']] 
    Register_df["grade"].replace({"Baby1":'Baby','Class5':'Grade5',}, inplace=True)
   

    fee_df[['Admission', 'Tuition',
        'Transport', 'Uniform', 'Lunch', 'Exams', 'BookLvy', 'Activity',
        'OtheLvy']] = fee_df[['Admission', 'Tuition',
        'Transport', 'Uniform', 'Lunch', 'Exams', 'BookLvy', 'Activity',
        'OtheLvy']].fillna(0)
    fee_df["total_paid"] =fee_df["Admission"] +fee_df["Tuition"] +fee_df["Transport"] +fee_df["Uniform"] \
        +fee_df["Lunch"] +fee_df["Exams"] +fee_df["BookLvy"] +fee_df["Activity"] +fee_df["OtheLvy"] 
    fee_df['id']=fee_df['id'].astype(object)
    fee_df['ReceiptNo']=fee_df['ReceiptNo'].astype(object)
    fee_df['DOP']=pd.to_datetime(fee_df['DOP']).dt.strftime('%Y-%m-%d')
    fees_df=pd.merge(left=fee_df, right=Register_df, how='left', left_on='ClassRegisterID', right_on='id')
    fees_df = fees_df.rename(columns=str.lower)
    fees_df.drop(["id_y",'classregisterid'], axis=1, inplace=True)
    fees_df.rename(columns = {'id_x':'id', }, inplace = True)
    fees_df=fees_df[['id', 'receiptno', 'dop', 'year', 'term', 'grade','adm','name_stud', 'enrolstatus', 'phone1', 'phone2', 'phone3', 'admission', 'tuition', 'transport',
        'uniform', 'lunch', 'exams', 'booklvy', 'activity', 'othelvy',  'total_paid']]


    fee_df1=fee_df.groupby(["ClassRegisterID"])["total_paid"].sum().reset_index(name='Total_collection')
    fees_bal_df=pd.merge(left=fee_df1, right=Register_df, how='outer', left_on='ClassRegisterID', right_on='id')
    fees_bal_df.Total_collection=fees_bal_df.Total_collection.fillna(0)
    fees_bal_df["bal_cf"] =fees_bal_df["tot_invoice"] - fees_bal_df["Total_collection"] 
    fees_bal_df=fees_bal_df[['id','year', 'term','grade',  'adm','name_stud','enrolstatus',  'bal_bf', 'tot_invoice', 'Total_collection','bal_cf' , 'phone1', 'phone2', 'phone3']]
    fees_bal_df = fees_bal_df.rename(columns=str.lower)


    exam_df=pd.merge(left=exams_df, right=Register_df, how='left', left_on='ClassRegisterID', right_on='id')
    exam_df.drop(["id_y",'ClassRegisterID'], axis=1, inplace=True)
    exam_df.rename(columns = {'id_x':'id', }, inplace = True)
    exam_df=exam_df[['id', 'ExamType', 'year', 'term', 'grade','adm',   'name_stud', 'dob', 'sex','enrolstatus',  'Maths', 'EngLan', 'EngComp',
        'KisLug', 'KisIns', 'Social', 'Creative', 'CRE', 'Science', 'HmScie',
        'Agric', 'Music', 'PE']]
    exam_df= exam_df.rename(columns=str.lower)
    subjects=['maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative',
        'cre', 'science', 'hmscie', 'agric', 'music', 'pe']
    exam_df['Tot_marks'] = exam_df[subjects].sum(numeric_only=True, axis=1)
    exam_df=exam_df.dropna()
    exam_df['year'] = exam_df.year.astype(int) 
    exam_df['adm'] = exam_df.adm.astype(int)  
        
    html_temp1 = """
    <div style="background-color:white;padding:1.5px">
    <h1 style="color:black;text-align:center;">JAVIC JUNIOR SCHOOL </h1>
    </div><br>"""

    html_temp2 = """
    <div style="background-color:white;padding:1.5px">
    <h3 style="color:black;text-align:center;">Management Mornitoring Application </h3>
    </div><br>"""
    st.markdown(html_temp1,unsafe_allow_html=True)
    _,_,_, col2, _,_,_ = st.beta_columns([1,1,1,2,1,1,1])
    #with col2:
    #st.image(im, width=150)

    st.markdown(html_temp2,unsafe_allow_html=True)
    #st.title('This is for a good design')
    st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)


    Infos="""
    Welcome to Javic Junior  School. You can use this application to monitor your child's fee and academic records. 
    For more information contact the director/Manager at 0710332465/0733332465/0710945685
    """
    st.markdown(Infos)

    selection = st.sidebar.selectbox('Select option to view:', 
        ('My Child', 'General School Information'))
    if selection == 'My Child':
        Phone =st.text_input(label='Enter your phone Number registered at school')
        if Phone=="":
            st.write('Enter your phone Number registered at school in the space above to continue')
        if Phone !="":
            st.write('These are the kids registered with the phone number provided')
            stud_df=stud_df[['ADM', 'name',  'Mother', 'Father', 'Guadian', 'DOB', 'sex','PHONE1', 'PHONE2', 'PHONE3', 'DOA']]
            st.dataframe(stud_df[((stud_df.PHONE1==Phone)| (stud_df.PHONE2==Phone)|(stud_df.PHONE3==Phone))].T)
            st.markdown("<h5 style='text-align: center; color: blue;'>--------------------------------------------------------------------------------------------------------------------</h5>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; color: red;'>If you cant see one or more of your child, try entering your phone number without the starting zero e.g 710945685 OR contact the office to update your phone contacts</h5>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; color: blue;'>--------------------------------------------------------------------------------------------------------------------</h5>", unsafe_allow_html=True)
            
            
            if st.checkbox("Show Account Status"):
                fees_bal_df=fees_bal_df[((fees_bal_df.enrolstatus=='In_Session')&((fees_bal_df.phone1==Phone)| (fees_bal_df.phone2==Phone)|(fees_bal_df.phone3==Phone)))]
                fees_bal_df.rename(columns = {'year':'Year', 'term':'Term', 'grade':'Class', 'adm':'ADM', 'name_stud':'Name', 'bal_bf':'Arrears', 'tot_invoice':'Fees', 'total_collection':'Paid', 'bal_cf':'Bal'}, inplace = True)
                fees_bal_df.drop(['id',"enrolstatus",'phone1','phone2','phone3'], axis=1, inplace=True)

                feesBal_select = st.radio( "How do you want to view account status", ('All accounts Status records', 'Filter Account Status Records by Year, Tearm and ADM'))
                if feesBal_select == 'All accounts Status records':
                    fees_bal_df = fees_bal_df.sort_values(['Year','Term','ADM'], ascending = (True, True,True))
                    st.dataframe(fees_bal_df[['Year','Term','ADM', 'Name', 'Arrears', 'Fees', 'Paid','Bal']])
                if feesBal_select == 'Filter Account Status Records by Year, Tearm and ADM':
                    yr = st.slider("Select Year", min_value=2020, max_value=2030, value=2020, step=1)
                    tm = st.selectbox("Select Term",options=['Term 1' , 'Term 2', 'Term 3'])
                    ADM = st.number_input('Enter Admission Number: (You can check the admission mumber from the student details above)', value = 0)
                    st.dataframe(fees_bal_df[((fees_bal_df.Year==yr)&(fees_bal_df.Term==tm)&(fees_bal_df.ADM==ADM))][['Year','Term','ADM', 'Name', 'Arrears', 'Fees', 'Paid','Bal']].T)


            if st.checkbox("Show Fee Payment History"):
                fees_df=fees_df[((fees_df.enrolstatus=='In_Session')&((fees_df.phone1==Phone)| (fees_df.phone2==Phone)|(fees_df.phone3==Phone)))]
                fees_select = st.radio( "How to view", ('All fee payment records', 'Filter fee payments - (view by year, term,  adm)'))
                if fees_select == 'All fee payment records':
                    st.dataframe(fees_df[['receiptno', 'dop', 'grade', 'adm', 'name_stud', 'total_paid']])
                if fees_select == 'Filter fee payments - (view by year, term,  adm)':
                    yr = st.slider("Select Year", min_value=2020, max_value=2030, value=2020, step=1)
                    tm = st.selectbox("Select Term",options=['Term 1' , 'Term 2', 'Term 3'])
                    ADM = st.number_input('Enter Admission Number:', value = 0)
                    st.dataframe(fees_df[((fees_df.year==yr)&(fees_df.term==tm)&(fees_df.adm==ADM))][['receiptno', 'dop', 'grade', 'adm', 'name_stud', 'total_paid']])   
            

        
    elif selection == 'General School Information':
        Register_df['grade'] = pd.Categorical(Register_df['grade'], ['Baby', 'PP1', 'PP2','Grade1',  'Grade2', 'Grade3', 'Grade4', 'Grade5'])        
        st.title("Nothing to view yet")
        

        

        
            
if __name__ =='__main__':
    main() 


#@st.cache

#st.balloons()


