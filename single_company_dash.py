import all the packages we need
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import plotly
import plotly.express as px
import time
import datetime
from datetime import datetime
#___________________________________________________Function Defines___________________________________________________________________#
#Function Defines
def date2qtr(x):
    QTR=(x.month-1)//3+1
    QTR="Q"+str(QTR)
    return QTR

def date2year(x):
    return x.year

def GM_format(x):
    if(x>=0.4):
        y= 'background-color: green'
    elif(x>=0):
        y='background-color: orange'
    else:
        y='background-color: red'
    return y

def SGA_format(x):#same as liability format
    if( (x<=0.8) and (x>0) ):
        y= 'background-color: green'
    else:
        y=''
    return y

def Interest_format(x):
    if(x>0.15):
        y= 'background-color: red'
    else:
        y=''
    return y

def Debt_format(x):
    if(x>=1):
        y= 'background-color: red'
    else:
        y=''
    return y

def Income_format(x):
    if(x>=0.15):
        y= 'background-color: green'
    else:
        y=''
    return y

def PE_format(x):
    if( (x<=40) and (x>0) ):
        y= 'background-color: green'
    else:
        y=''
    return y

def Ret_Earnings_format(x):
    if(x>=0.05):
        y= 'background-color: green'
    else:
        y=''
    return y
def Stock_inc_format(x):
    if(x>=0.03):
        y= 'background-color: green'
    else:
        y=''
    return y
#force the layout to be wide
st.set_page_config(layout="wide")
#______________________________________________S&P500________________________________________________________________________________#
#Grab SP500 data
SP500=yf.Ticker('SPY').history(period='max', interval = "3mo")
SP500=SP500.dropna()
year=[]
qrt=[]
increase=[]
prev_value=SP500['Close'][0]
for row in SP500.itertuples():
    year.append(row.Index.year)
    qrt.append(date2qtr(row.Index))
    increase.append((row.Close-prev_value)/prev_value)
    prev_value=row.Close

SP500['Year']=year
SP500['QTR']=qrt
SP500['SP500 Increase']=increase
SP500_classifier = pickle.load(open('ML_fundamental_analysis_model.sav', 'rb'))

#__________________________________Load the Data set, filter on ticker________________________________________________________________#
companyDF=pd.read_csv(r'complete_financial_information.csv')
company_list=companyDF.drop_duplicates(subset='Ticker')

company_list=company_list['Ticker'].values
#st.title('Stock Dashboard')

cols = st.beta_columns(2)

company_ticker = cols[0].selectbox('Ticker:',company_list)
filtered_data = companyDF[companyDF['Ticker'] == company_ticker]

#need to sort the data
#need to save a copy becasue the sort screws up the plots
plot_data=filtered_data
filtered_data=filtered_data.sort_values(by=['QTR'], ascending=False, kind='mergesort')
filtered_data=filtered_data.sort_values(by=['Year'], ascending=False, kind='mergesort')

#__________________________________Grab the data for plotting________________________________________________________________#

tempDF=filtered_data[['Profit_Margin','SGA_pct_Profit','Interest_pct_Income',
        'Income_pct_Revenue','Earnings_pct_per_share_price','Debt_pct_Cash','ROI',
        'Earnings_pct_Equity','Liabilites_per_Equity_plus_Tresury Stock','PE','Equity_per_share Price',
        'Retained_Earnings_Increase','Income_Increase','Stock_Increase']].copy()
#Sets whether we show all the data or just the main stuff
data_style=cols[1].radio('Summary or Full Data',['Summary','Full'])
if(data_style=='Full'):
    new_temp=filtered_data
else:
    new_temp=filtered_data[['Report_Date','Profit_Margin','SGA_pct_Profit','Interest_pct_Income',
        'Income_pct_Revenue','Debt_pct_Cash', 'Liabilites_per_Equity_plus_Tresury Stock','PE',
        'Retained_Earnings_Increase','Income_Increase','Stock_Increase']].copy()#,'ROI','Equity_per_share Price',
new_style=(new_temp.style
               .applymap(GM_format, subset=['Profit_Margin'])
               .applymap(SGA_format, subset=['SGA_pct_Profit','Liabilites_per_Equity_plus_Tresury Stock'])
               .applymap(Interest_format, subset=['Interest_pct_Income'])
               .applymap(Income_format, subset=['Income_pct_Revenue'])
               .applymap(PE_format, subset=['PE'])
               .applymap(Ret_Earnings_format, subset=['Retained_Earnings_Increase'])
               .applymap(Stock_inc_format, subset=['Stock_Increase'])
               .applymap(Debt_format, subset=['Debt_pct_Cash'])
          )#.format("{:.2%}")) 

st.dataframe(new_style, height=500)
#__________________________________Run the ML model________________________________________________________________#
tempX=tempDF.iloc[:, 0:14].values
temp_predict=SP500_classifier.predict_proba(tempX)
temp_predict=temp_predict[:,1]
#__________________________________Create the plot of Stock and vs SP500___________________________________________________#
#Add a way to change the start date of the simulation against the SP500
temp_company = plot_data.iloc[1: , :]
SP_inv=100
comp_inv=100
SP=[]
comp=[]
SP.append(SP_inv)
comp.append(comp_inv)
for row in temp_company.itertuples(): 
    SPtemp=SP500[SP500['Year']==row.Year]
    SPtemp=SPtemp[SPtemp['QTR']==row.QTR]
    SP_inc=SPtemp.iloc[0,9]
    comp_inc=row.Stock_Increase
    SP_inv=SP_inv*(1+SP_inc)
    comp_inv=comp_inv*(1+comp_inc)
    SP.append(SP_inv)
    comp.append(comp_inv)
#print(max(np.array(SP)))
max1=max(np.array(SP))
max2=max(np.array(comp))
max1=max(max1,max2)
SP=SP/max1
comp=comp/max1
plot2DF=pd.DataFrame(comp, columns=[company_ticker])
plot2DF['SP500']=SP
plot2DF['Date']=plot_data['Report_Date'].values
plot2DF['Date']=pd.to_datetime(plot2DF['Date'])
plot2DF['ML Rating']=temp_predict
fig2 = px.line(plot2DF,x="Date", y=[company_ticker, 'SP500', 'ML Rating'],)# title='SP500 Compare')

cols[0].plotly_chart(fig2, use_container_width=True)
#__________________________________Create the plot of Stock and ML output___________________________________________________#
treasury=plot_data['Treasury_Stock'].values*(-1)
RandD=plot_data['RandD'].values*(-1)
max1=max(np.array(RandD))
max2=max(np.array(treasury))

#treasury=treasury/max2
#RandD=RandD/max1
plot1DF=pd.DataFrame(treasury,columns=['Treasury Stock'])
plot1DF['Date']=plot_data['Report_Date'].values
plot1DF['Date']=pd.to_datetime(plot1DF['Date'])
plot1DF['R&D']=RandD
fig1 = px.line(plot1DF, x="Date", y=["Treasury Stock", 'R&D'],)# title='ML Output vs. Stock Prce')
cols[1].plotly_chart(fig1, use_container_width=True)
