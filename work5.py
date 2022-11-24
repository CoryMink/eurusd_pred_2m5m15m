# Libraries
import time
import pandas    as pd
import numpy     as np
import datetime  as dt
import yfinance  as yf
import pandas_ta as ta
import streamlit as st
import plotly.graph_objects  as go
from streamlit_autorefresh   import st_autorefresh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.linear_model    import Ridge

    
# Streamlit showtime
st_autorefresh(interval=30000, key='prediction')
st.title("EURUSD 💶💵 Prediction")

tik = st.selectbox('Select the currency pair',
                   ("EURUSD=x","USDJPY=x","GBPUSD=x"))

# Predictions ======================================================
tickers   = tik                 # symbol based on yfinance
intervals = ['2m','5m','15m']   # time intervals
y_preds   = []                  # blank prediction list
for i in intervals:
    # download data
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(minutes=50000)
    data = yf.download(tickers=tickers,interval=i, 
                       start=startdate, end=enddate,progress=False)
    # adding indicators
    data['ROC']  = ta.roc(data.Close, length=21)
    data['EMAM'] = ta.ema(data.Close, length=100)
    data['EMAS'] = ta.ema(data.Close, length=200)
    data['WILL'] = ta.willr(data.High,data.Low,data.Close)
    data['BOP']  = ta.bop(data.Open,data.High,data.Low,data.Close)
    data['PDIS'] = ta.pdist(data.Open,data.High,data.Low,data.Close)
    data['EBSW'] = ta.ebsw(data.Close, length=21)
    data['Poly1'] = data['BOP'] * data['PDIS'] * data['WILL']
    data['ATR']   = ta.atr(data.High,data.Low,data.Close,length=14,mamode='SMA')
    data['NextClose'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    X = data[['Open','High','Low','ROC','EMAM','EMAS',
              'WILL','BOP','PDIS','EBSW','Poly1','ATR']]
    y = data['NextClose']
    X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=0.1)
    sc = MinMaxScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)
    rg = Ridge(alpha=0.0001, solver='cholesky')
    rg.fit(X_train_sc,y_train)
    y_pred = rg.predict(X_test_sc)
    y_preds.append(round(y_pred[-1],6))
# =============================================================== #
    
col1, col2 = st.columns(2, gap='small')

with col1:
    try:
        st.subheader('Before Trading')
        cur_total = st.number_input('Current Total')
        profit    = st.number_input('Curret Profit')
        bid       = round(cur_total*0.05/profit*100,2)
        st.write('Bid: ',bid)
    except ZeroDivisionError:
        st.error('Please input your information', icon='🚩')


with col2:
    st.subheader('Prediction')
    st.write('')
    st.write('')
    st.table(pd.Series(y_preds,index=intervals))

enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(minutes=450)
data = yf.download(tickers=tik,interval='1m', 
                   start=startdate, end=enddate,progress=False)
def support(df,l,n1,n2):

    for i in range(l-n1+1,l+1):
        if(df['Low'][i]>df['Low'][i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df['Low'][i]<df['Low'][i-1]):
            return 0
    return 1

def resistance(df,l,n1,n2):

    for i in range(l-n1+1,l+1):
         if(df['High'][i]>df['High'][i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df['High'][i]<df['High'][i-1]):
            return 0
    return 1

ss = []
rr = []
n1 = 2 
n2 = 2 
for row in range(5,len(data)-n2):
    if support(data,row,n1,n2):
        ss.append((row,data.Low[row],1))
    if resistance(data,row,n1,n2):
        rr.append((row,data.High[row],2))

sslist = [x[1] for x in ss if x[2]==1]
rrlist = [x[1] for x in rr if x[2]==2]
sslist.sort()
rrlist.sort()

for i in range(1,len(sslist)):
    if(i>=len(sslist)):
        break
    if abs(sslist[i]-sslist[i-1])<=0.005:
        sslist.pop(i)

for i in range(1,len(rrlist)):
    if(i>=len(rrlist)):
        break
    if abs(rrlist[i]-rrlist[i-1])<=0.005:
        rrlist.pop(i)        
fig = go.Figure(data=go.Candlestick(x=data.index,
                        open=data.Open,
                        high=data.High,
                        low=data.Low,
                        close=data.Close))
for i in y_preds:
    fig.add_hline(y=i, line_color="black", line_dash="dash")
for s in sslist:
    fig.add_hline(y=s, line_color="green", opacity=0.5)
for r in rrlist:
    fig.add_hline(y=r, line_color="red", opacity=0.5)

st.plotly_chart(fig)
st.write('Resistance: 🟥 | Support: 🟩')
st.write()
st.write('Target:     ⬛')