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
    session_df = pd.read_csv(class_session)
    classregister_df = pd.read_csv(classregister)
    invoice_df = pd.read_csv(invoice)
    employees_df = pd.read_csv(employees)
    fee_df = pd.read_csv(fees)
    exams_df = pd.read_csv(exams)

    session = pd.merge(left=session_df, right=employees_df[['id', 'name', 'Staff_ID', 'sex']], how='left', left_on='ClassTeacher', right_on='id')
    session.drop(["id_y","ClassTeacher"], axis=1, inplace=True)
    session.rename(columns = {'name':'ClassTeacher','id_x':'id', }, inplace = True)
    #session
    Register = pd.merge(left=classregister_df, right=session, how='left', left_on='Session', right_on='id')
    Register.drop(["id_y","Session"], axis=1, inplace=True)
    Register.rename(columns = {'id_x':'id', }, inplace = True)

    Register = pd.merge(left=Register, right=stud_df, how='left', left_on='student', right_on='ID')
    Register.drop(['ID','Mother', 'Father', 'Guadian',
        'Class_Admitted', 'PHONE1', 'PHONE2', 'PHONE3', 'DOA', 'address',
            'DateExit',"student",'Startdate','Enddate'], axis=1, inplace=True)
    Register.rename(columns = {'sex_x':'sex_teacher','sex_y':'sex_stud' ,'name':'name_stud' ,'Class_grd':'grade'  }, inplace = True)

    Register=pd.merge(left=Register, right=invoice_df, how='left', left_on='id', right_on='ClassRegisterID')
    Register.drop(["id_y"], axis=1, inplace=True)
    Register.rename(columns = {'id_x':'id', }, inplace = True)
    Register_df = Register.rename(columns=str.lower) 

    Register_df['tot_invoice']=Register_df['bal_bf']+Register_df['uniform']+Register_df['uniform_no']+(Register_df['transport']*Register_df['transport_months'])+ \
        (Register_df['lunch']*Register_df['lunch_months'])+Register_df['otherlevyindv']+Register_df['tutionfee']+Register_df['examfee']+ \
            Register_df['booklevy'] +Register_df['activityfee']+Register_df['otherlevies']
    Register_df= Register_df[['id', 'year', 'term',"classregisterid", 'grade', 'classteacher', 'staff_id', 'sex_teacher',
        'name_stud', 'adm', 'dob', 'sex_stud', 'enrolstatus', 'bal_bf', 'tot_invoice']]      
    Register_df["grade"].replace({"Baby1":'Baby','Class5':'Grade5',}, inplace=True)
    
    #Register_df.to_csv(r'API\data\Register_df.csv', index = False)
    #prepare fee data 
    fee_df[['Admission', 'Tuition',
        'Transport', 'Uniform', 'Lunch', 'Exams', 'BookLvy', 'Activity',
        'OtheLvy']] = fee_df[['Admission', 'Tuition',
        'Transport', 'Uniform', 'Lunch', 'Exams', 'BookLvy', 'Activity',
        'OtheLvy']].fillna(0)
    fee_df["total_paid"] =fee_df["Admission"] +fee_df["Tuition"] +fee_df["Transport"] +fee_df["Uniform"] \
        +fee_df["Lunch"] +fee_df["Exams"] +fee_df["BookLvy"] +fee_df["Activity"] +fee_df["OtheLvy"] 
    fee_df['id']=fee_df['id'].astype(object)
    fee_df['ReceiptNo']=fee_df['ReceiptNo'].astype(object)
    fee_df['DOP1']=pd.to_datetime(fee_df['DOP'] ,format = '%Y-%m-%d') 
    fee_df['DOP']=pd.to_datetime(fee_df['DOP']).dt.strftime('%Y-%m-%d')
    fees_df=pd.merge(left=fee_df, right=Register_df, how='left', left_on='ClassRegisterID', right_on='classregisterid')
    fees_df = fees_df.rename(columns=str.lower)
    fees_df.drop(["id_y",'classregisterid'], axis=1, inplace=True)
    fees_df.rename(columns = {'id_x':'id', }, inplace = True)
    fees_df=fees_df[['id', 'receiptno', 'dop', 'year', 'term', 'grade','adm','name_stud', 'enrolstatus', 'admission', 'tuition', 'transport',
        'uniform', 'lunch', 'exams', 'booklvy', 'activity', 'othelvy',  'total_paid','dop1']]
    fees_df["pay_year"] = fees_df.dop1.dt.year 
    fees_df["pay_month"] = fees_df.dop1.dt.month 
    fees_df["pay_day"] = fees_df.dop1.dt.day
    fees_df['year']=  fees_df['year'].apply(str)
    fees_df['pay_year']=  fees_df['pay_year'].apply(str)
    #fees_df["pay_month"].replace({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec' }, inplace=True)
    

    fee_df1=fee_df.groupby(["ClassRegisterID"])["total_paid"].sum().reset_index(name='Total_collection')
    fees_bal_df=pd.merge(left=fee_df1, right=Register_df, how='left', left_on='ClassRegisterID', right_on='classregisterid')
    fees_bal_df["bal_cf"] =fees_bal_df["tot_invoice"] - fees_bal_df["Total_collection"] 
    fees_bal_df=fees_bal_df[['ClassRegisterID','year', 'term','grade',  'adm','name_stud','enrolstatus',  'bal_bf', 'tot_invoice', 'Total_collection','bal_cf']]
    fees_bal_df = fees_bal_df.rename(columns=str.lower)
    fees_bal_df['year']=  fees_bal_df['year'].apply(str)
    
    #Exams 
    exam_df=pd.merge(left=exams_df, right=Register_df, how='left', left_on='ClassRegisterID', right_on='classregisterid')
    exam_df.drop(["id_y",'classregisterid'], axis=1, inplace=True)
    exam_df.rename(columns = {'id_x':'id', }, inplace = True)
    exam_df=exam_df[['id', 'ClassRegisterID', 'ExamType', 'year', 'term', 'grade', 'classteacher','adm',   'name_stud', 'dob', 'sex_stud','enrolstatus',  'Maths', 'EngLan', 'EngComp',
        'KisLug', 'KisIns', 'Social', 'Creative', 'CRE', 'Science', 'HmScie',
        'Agric', 'Music', 'PE']]
    exam_df = exam_df.rename(columns=str.lower)
    exam_df['year'] = pd.to_numeric(exam_df['year'], errors='coerce')
    exam_df['adm'] = pd.to_numeric(exam_df['adm'], errors='coerce')
    subjects=['maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative',
       'cre', 'science', 'hmscie', 'agric', 'music', 'pe']
    exam_df['Tot_marks'] = exam_df[subjects].sum(numeric_only=True, axis=1)
    exam_df=exam_df.dropna()
    exam_df['year'] = exam_df.year.astype(int) 
    exam_df['adm'] = exam_df.adm.astype(int)  

    ####################
    ML_df=pd.merge(left=exam_df, right=fee_df, how='right', left_on='classregisterid', right_on='ClassRegisterID')
    ML_df.drop(["id_y",'examtype','enrolstatus','ClassRegisterID','ReceiptNo', 'DOP','DOP1'], axis=1, inplace=True)
    ML_df.rename(columns = {'id_x':'id', }, inplace = True)

    ML_df=pd.merge(left=ML_df, right=fees_bal_df, how='right', left_on='classregisterid', right_on='classregisterid')
    ML_df.drop(['year_y', 'term_y', 'grade_y','Admission', 'Tuition', 'Uniform', 'Exams', 'BookLvy', 'Activity', 'OtheLvy',
        'total_paid','id', 'adm_x',  'name_stud_x','enrolstatus','maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative', 'cre',
        'science', 'hmscie', 'agric', 'music', 'pe','dob'], axis=1, inplace=True)
    ML_df.rename(columns = {'year_x':'year', 'term_x':'term', 'grade_x':'grade','sex_stud':'sex', 'name_stud_y':'name', 'adm_y':'adm' }, inplace = True)

    selection = st.sidebar.selectbox('Select your analysis option:', 
        ('Background Information', 'Data Intergration', 'Descriptive Analysis', 'Machine Learning'))
    if selection == 'Background Information':
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
        Welcome to Javic Junior  School. This application will be used to monitor the the financial 
        management operations and academics of our students. For more information contact the director @
        javicjun@javicjuniorschool.co.ke  or Manager @  vogendo@javicjuniorschool.co.ke
        """
        st.markdown(Infos)
        

        st.markdown("**Project Problem Statement**")
        Problem="""
        Javic Junior School is a private primary and pre-primary school based in Kisumu, 
        Kenya. It is a partnership business between Dr. Jacob Ong'ala and Mrs. Vivian Awuor
        The two partners stay apart (in different Countries) but they participarte in the daily, 
        Operations of the school. And thefore the need for an automated online management 
        plattform 

        To make decisions as managers, they get alot of reference from the data that they collect 
        from School operations. In July 2020, Javic Junior School upgraded their data management system to a 
        web-based (www.javicjuniorschool.com). The system is hosted by a web company and is runing on MySQL laguange. 

        The management would like to use the data to make decision in aswering the following 
        following question a
        """

        st.markdown(Problem)
        st.markdown("* Study the student population (interm of gender, grade, term and year)")
        st.markdown("* Monitor fee collection  and establish fee balance status")
        st.markdown("* Report on students perormance  ")
        st.markdown("* Predict students performances based on other collected features ")
        

        st.markdown("**Methodology**")
        methods="""
        The database is hosted in a webhosting company but can be accessed in form of variou csv tables. They will 
        be exctracted from the database and saved github repository where the application can access them easily 
        for management and building model. 

        Since the table are from arelational relational database, they will be marged accordingly to form dataframe(s) which will be used in the 
        application building
        """
        st.markdown(methods)


    elif selection == 'Data Intergration':
        Register_df['grade'] = pd.Categorical(Register_df['grade'], ['Baby', 'PP1', 'PP2','Grade1',  'Grade2', 'Grade3', 'Grade4', 'Grade5'])        
        st.title("Cleaned/Merged Data")
        st.markdown("**Select the data set that you want to view**")
        if st.checkbox("Show Students Dataset"):
            #st.write('Students data')
            st.dataframe(Register_df[Register_df.enrolstatus=='In_Session'].head(5))
        if st.checkbox("Show Fee Collection Dataset"):
            #st.write('Feee data')
            st.dataframe(fees_df[fees_df.enrolstatus=='In_Session'][['receiptno', 'dop', 'grade', 'adm', 'name_stud',
            'admission', 'tuition', 'transport', 'uniform', 'lunch',
            'exams', 'booklvy', 'activity', 'othelvy', 'total_paid']])    
        if st.checkbox("Show Fee balances Dataset"):
            #st.write('Feee data')
            st.dataframe(fees_bal_df[fees_bal_df.enrolstatus=='In_Session'][['year', 'term', 'grade', 'adm', 'name_stud',
            'bal_bf', 'tot_invoice', 'total_collection', 'bal_cf']])   
        if st.checkbox("Show Academic Dataset"):
            #st.write('Feee data')
            st.dataframe(exam_df[exam_df.enrolstatus=='In_Session'][['examtype', 'year', 'term', 'grade',
            'adm', 'name_stud', 'maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative',
            'cre', 'science', 'hmscie', 'agric', 'music', 'pe']]) 
    
    elif selection == 'Descriptive Analysis':
        st.title("Descriptive Analysis")
        if st.checkbox("Student Population"):
            curent_pop=Register_df.drop_duplicates('adm').id.count()
            st.write("the current student population is ", curent_pop)
            stud_select = st.radio( "Select statistics to view", ('Population by Class', 'Population by Gender'))
            if stud_select == 'Population by Class': 
                Register_df['grade'] = pd.Categorical(Register_df['grade'], ['Baby', 'PP1', 'PP2','Grade1',  'Grade2', 'Grade3', 'Grade4', 'Grade5'])    
                a = Register_df.groupby(["term",'grade', 'sex_stud'])["id"].count().reset_index(name='Number')
                fig = px.bar(a, x='grade', y='Number',color='term',barmode='group',hover_name='sex_stud',text='Number',)
                st.plotly_chart(fig)
                
            if stud_select == 'Population by Gender': 
                df = Register_df.groupby(["term",'sex_stud'])["id"].count().reset_index(name='Number')
                # Create subplots: use 'domain' type for Pie subplot
                fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
                fig.add_trace(go.Pie(labels=df[df.term== 'Term 2'].sex_stud.array, values=df[df.term== 'Term 2'].Number.array, name="Term 2"),
                            1, 1)
                fig.add_trace(go.Pie(labels=df[df.term== 'Term 3'].sex_stud.array, values=df[df.term== 'Term 3'].Number.array, name="Term 3"),
                            1, 2)
                fig.update_traces(hole=.4, hoverinfo="label+value+name")
                fig.update_layout(
                    title_text="Population by gender",
                    # Add annotations in the center of the donut pies.
                    annotations=[dict(text='Term 2', x=0.18, y=0.5, font_size=20, showarrow=False),
                                dict(text='Term 3', x=0.82, y=0.5, font_size=20, showarrow=False)])
                st.plotly_chart(fig)
        if st.checkbox("Fee Collection Statistics"):
            fees_select = st.radio( "Choose fee collection data to view", ('monthly collection', 'Termly collection', 'Collection by Class', 'Number of payments per day'))
            if fees_select == 'monthly collection': 
                #fees_df['pay_year']= pd.Categorical(fees_df['pay_year'])
                #fees_df['pay_month'] = pd.Categorical(fees_df['pay_month'], ['Jan', 'Feb', 'Mar','Apr',  'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 
                #fees_df.sort_values('pay_month',inplace=True, ascending=True)
                fees_df['pay_month'] = fees_df['pay_month'].replace([1, 2,3,4,5,6,7,8,9,10,11,12],   ['Jan', 'Feb', 'Mar','Apr',  'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 

                # st.dataframe(pd.crosstab(fees_df.pay_year,  fees_df.pay_month))
                a = fees_df.groupby(["pay_year",'pay_month'])["total_paid"].sum().reset_index(name='Monthl_fee_received')
                fig = px.bar(a, x='pay_month', y='Monthl_fee_received',color='pay_year',barmode='group',text='Monthl_fee_received',
                labels=dict(pay_month="Month", pay_year="Year", Monthl_fee_received="Fee received (Ksh)"))
                fig.update_xaxes(type='category')
                st.plotly_chart(fig)
            if fees_select == 'Termly collection': 
                #st.dataframe(pd.crosstab(fees_df.year,  fees_df.term))
                a = fees_df.groupby(["year",'term'])["total_paid"].sum().reset_index(name='Termly_fee_received')
                fig = px.bar(a, x='term', y='Termly_fee_received',color='year',barmode='group',text='Termly_fee_received',
                labels=dict(term="Term", year="Year", Termly_fee_received="Fee received (Ksh)"))
                st.plotly_chart(fig)
            if fees_select == 'Collection by Class':
                #st.dataframe(pd.crosstab(fees_df.year,  fees_df.term))
                a = fees_df.groupby(["year",'grade', 'term' ])["total_paid"].sum().reset_index(name='Fee_by_grade')
                fig = px.bar(a, x='grade', y='Fee_by_grade',color='year',facet_row='term', barmode='group', text='Fee_by_grade',
                labels=dict(grade="Grade", year="Year", Fee_by_grade="Fee received (Ksh)", term='Term'))
                st.plotly_chart(fig)
            if fees_select == 'Number of payments per day':
                #st.dataframe(pd.crosstab(fees_df.year,  fees_df.term))
                a = fees_df.groupby(['pay_day','pay_month'])["receiptno"].count().reset_index(name='counts')
                a = a.groupby(['pay_day'])["counts"].mean().reset_index(name='Average')
                fig = px.line(a, x='pay_day', y='Average', 
                labels=dict(pay_day="Day of the month", Average="Average number of Payments"))
                st.plotly_chart(fig)
        if st.checkbox("Fee balance Statistics"):
            fees_select = st.radio( "Choose fee Balances Information to view", ('Fee Balances Table', 'Fee balances Charts', 'Individual Fee Balance', 'Bad debts'))
            if fees_select == 'Fee Balances Table':
                fees_bal_df=fees_bal_df[fees_bal_df.enrolstatus =='In_Session']
                fees_bal_df=fees_bal_df.drop(['classregisterid', 'enrolstatus'], axis=1).sort_values(['year', 'term', 'grade','name_stud', 'adm' ], ascending = (False, False, True, True, False))
                st.dataframe(fees_bal_df)
               
            if fees_select == 'Fee balances Charts':
                
                a = fees_bal_df.groupby(['year', 'term', 'grade',])[["bal_bf", "tot_invoice" ,"bal_cf" ]].sum().reset_index()
                fig = px.bar(a, x='grade', y='bal_bf',color='year',facet_col='term', barmode='group', text='bal_bf',
                labels=dict(grade="Grade", year="Year", bal_bf="Fee Balance (Ksh)", term='Term'))
                fig2 = px.bar(a, x='grade', y='bal_cf',color='year',facet_col='term', barmode='group', text='bal_cf',
                labels=dict(grade="Grade", year="Year", bal_cf="Fee Balance (Ksh)", term='Term'))
                st.plotly_chart(fig)
                st.plotly_chart(fig2)
           
               
            if fees_select == 'Individual Fee Balance':
                ADM = st.number_input('Enter Admission Number:', value = 0)
                fees_bal_df=fees_bal_df.loc[fees_bal_df['adm'] == ADM]
                fees_bal_df=fees_bal_df.drop(['classregisterid', 'enrolstatus'], axis=1).sort_values(['bal_cf', 'year'], ascending = (False,False))
                st.dataframe(fees_bal_df)     
               
               
            if fees_select == 'Bad debts':
                fees_bal_df=fees_bal_df.loc[(fees_bal_df['enrolstatus'] == 'exits') & fees_bal_df['bal_cf'] >0]
                fees_bal_df=fees_bal_df.drop(['classregisterid', 'enrolstatus'], axis=1).sort_values(['bal_cf', 'year'], ascending = (False,False))
                st.dataframe(fees_bal_df)
        if st.checkbox("Academic Statistics"):
            exam_df=exam_df.drop(['id', 'classteacher',  'dob', 'sex_stud'], axis=1).dropna()
            exam_df=exam_df[exam_df.enrolstatus =='In_Session']
            yr = st.slider("Select Year", min_value=2020, max_value=2030, value=2020, step=1)
            tm = st.selectbox("Select Term",options=['Term 1' , 'Term 2', 'Term 3'])
            grd = st.selectbox("Select Grade",options=['Baby', 'PP1', 'PP2', 'Grade1', 'Grade2', 'Grade3', 'Grade4', 'Grade5'])
            exam_df=exam_df[(exam_df.year==yr)& (exam_df.term ==tm) & (exam_df.grade==grd)]
            exam_df=exam_df.groupby(["adm", 'name_stud'])['maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative', 'cre', 'science', 'hmscie', 'agric', 'music', 'pe',
            'Tot_marks'].mean().reset_index().sort_values('Tot_marks', ascending = (False)) 
            exam_df[['maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative', 'cre', 'science', 'hmscie', 'agric', 'music', 'pe','Tot_marks']] = exam_df[['maths', 'englan', 'engcomp', 'kislug', 'kisins', 'social', 'creative', 'cre', 'science', 'hmscie', 'agric', 'music', 'pe', 'Tot_marks']].astype(int)
                    
            if grd in ['Baby', 'PP1', 'PP2']:
                exam_df['pos'] =exam_df.Tot_marks.rank(ascending=False).apply(np.floor).astype(int)
                exam_df.rename(columns = {'adm':'ADM','name_stud':'Name','maths':'Mathematics', 'englan':'Language', 'kislug':'Kiswahili',  'science':'Enviromental',  'cre':'Religious',  'creative':'Creative',   
                'Tot_marks':'Total'}, inplace = True)
                if grd in ['Baby', 'PP1']:
                    exam_df=exam_df[['ADM', 'Name', 'pos', 'Mathematics', 'Language', 'Creative', 'Religious', 'Enviromental', 'Total']]
                    st.dataframe(exam_df)       
                else :
                    exam_df=exam_df[['ADM', 'Name', 'pos', 'Mathematics', 'Language', 'Creative', 'Religious', 'Enviromental','Kiswahili', 'Total']]
                    st.dataframe(exam_df) 

            if grd in ['Grade1', 'Grade2', 'Grade3']:
                exam_df['pos'] =exam_df.Tot_marks.rank(ascending=False).apply(np.floor).astype(int)
                exam_df['Kiswahili'] =exam_df['kislug']+exam_df['kisins'];exam_df['English'] =exam_df['englan']+exam_df['engcomp']
                exam_df.rename(columns = {'adm':'ADM','name_stud':'Name','maths':'Mathematics',  'social':'Hygiene' ,'science':'Enviromental',  'cre':'CRE',  'creative':'Creative',   
                'Tot_marks':'Total'}, inplace = True)
                exam_df=exam_df[['ADM', 'Name', 'pos', 'Mathematics', 'English','Kiswahili','Hygiene', 'Enviromental','CRE', 'Creative', 'Total']]
                st.dataframe(exam_df)  

                
            if grd in ['Grade4']:
                exam_df['pos'] =exam_df.Tot_marks.rank(ascending=False).apply(np.floor).astype(int)
                exam_df['Kiswahili'] =exam_df['kislug']+exam_df['kisins'];exam_df['English'] =exam_df['englan']+exam_df['engcomp']
                exam_df.rename(columns = {'adm':'ADM','name_stud':'Name','maths':'Mathematics',  'social':'S/Studies' ,'science':'Science',  'cre':'CRE',  'creative':'Art&Craft',   
                'hmscie':'H/Science', 'agric':'Agric', 'music':'Music', 'pe':'PE','Tot_marks':'Total'}, inplace = True)
                exam_df=exam_df[['ADM', 'Name', 'pos', 'Mathematics', 'English','Kiswahili','Science','H/Science','Agric','Art&Craft', 'Music','S/Studies' ,'CRE', 'PE', 'Total']]
                st.dataframe(exam_df)     

            if grd in ['Grade5']:
                exam_df['pos'] =exam_df.Tot_marks.rank(ascending=False).apply(np.floor).astype(int)
                exam_df['Kiswahili'] =exam_df['kislug']+exam_df['kisins'];exam_df['English'] =exam_df['englan']+exam_df['engcomp']
                exam_df.rename(columns = {'adm':'ADM','name_stud':'Name','maths':'Mathematics',  'social':'S/Studies' ,'science':'Science','Tot_marks':'Total'}, inplace = True)
                exam_df=exam_df[['ADM', 'Name', 'pos', 'Mathematics', 'English','Kiswahili','Science','S/Studies' , 'Total']]
                st.dataframe(exam_df) 
    
    else :   
        st.title("Machine Learning ")
       
        if st.checkbox("Prediction of students performance"):
            st.write("Below is the Performance predictions of students ")
            model = RandomForestRegressor(n_estimators=160,
                                   min_samples_leaf=3,
                                    min_samples_split=16,
                                    max_features='auto',
                                    n_jobs=-1,
                                   random_state=42)
            def modelling_Javic(df):
                df=df.dropna()
                df['year'] = df.year.astype(int) 
                df['Max_marks'] =df['grade']
                df["Max_marks"].replace({"Baby":500,  'PP1':500,'PP2':600,'Grade1':700,'Grade2':700,'Grade3':700,'Grade4':1100,'Grade5':500}, inplace=True)
                df['performance'] =100*df['Tot_marks'] /df['Max_marks'] 
                df['bal_bf'] =100*df['bal_bf'] /df['tot_invoice']    
                df['total_collection'] =100*df['total_collection'] /df['tot_invoice']  
                df['bal_cf'] =100*df['bal_cf'] /df['tot_invoice'] 
                df= df.rename(columns=str.lower)
                df.loc[df['transport'] > 0, 'transport'] = 1
                df.loc[df['lunch'] > 0, 'lunch'] = 1
                df['transport'] = df.transport.astype(int) 
                df['lunch'] = df.lunch.astype(int) 
                df=df.drop_duplicates('classregisterid', keep='last')
                df=df[['name','adm','term', 'grade', 'classteacher', 'sex','transport', 'lunch', 'bal_bf', 'total_collection', 'bal_cf', 'performance']]
                df=df[(df.bal_bf > -50)&(df.bal_bf < 150)] #Avoiding very large of very small balance bf
                df=df[['name','adm','grade', 'sex','transport', 'lunch', 'bal_bf', 'total_collection', 'bal_cf', 'performance']]
                
                categorical = []
                for column in df.columns:
                    if df[column].dtypes != 'float64':
                        categorical.append(column)


                continuous  = []
                for column in df.columns:
                    if df[column].dtypes == 'float64':
                        continuous .append(column)
                continuous.remove('performance')

                threshold = 0.3705
                zscore = np.abs(stats.zscore(df[['bal_bf']]))
                df = df[(zscore > threshold).all(axis = 1)]
                df1=df.copy()
                encoded_features = {}
                for column in categorical:
                    encoded_features[column] = df.groupby([column])['performance'].median().to_dict()
                    df[column] = df[column].map(encoded_features[column])
                X = df.drop(['performance','adm','name'], axis = 1)
                model.fit(X, df['performance'])
                Y_pred = model.predict(X)
                df1['Predicted_performance'] = Y_pred
                predicted=df1[['adm','name','Predicted_performance']]
                return predicted
            predicted = modelling_Javic(ML_df)
            st.dataframe(predicted)

        

        
            
if __name__ =='__main__':
    main() 


#@st.cache

#st.balloons()


