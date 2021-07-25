# Imports
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io
import random
import datetime
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from SessionState import SessionState as session_state
import base64


# Streamlit Test Page
def build_page_test(session: session_state):
    st.header("This page helps you to simulate your CRM data")
        
    in_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type='csv')

            
    def plot(df, customerno):
        
        # df = pd.read_csv('sample.csv', delimiter=(';'))
        
    
        def makerandomnooforders(signal,customerno ):
            
            # p1 = customerno 
            total = 0
            max_no_order = (signal // customerno)
            nooforders = []
            for i in range(0, customerno):
                if total < signal:
                    tempval = random.randint(0, max_no_order)
                    nooforders.append(tempval)
                    total = total + tempval
                    leftover = signal - total
                elif total >= signal:
                    nooforders.append(0)
            if total < signal:
                #print('no')
                deltaval = signal - total
                nooforders[0] = nooforders[0] + deltaval
            elif total > signal:
                #print('yes')
                v = 0
                deltaval = total - signal
                while nooforders[v] < deltaval:
                        #print(  v)
                        #print(nooforders[v])
                        v +=1 
            if total > signal:
                nooforders[v] = nooforders[v] - deltaval
            #print(nooforders)
            return nooforders
        
        
        
        
        customeridls = []
        orderdatels = []
        customerdb = pd.DataFrame(columns=['customer_id', 'order_date','sale'])
        
        for h in range(customerno):
            customerid = 'C' + str(h)
            customeridls.append(customerid)
        
        for db in (dfls):
            # print(db)
        
            sale = (db['signal']).tolist()
            date = db['date'].tolist()
            # print(sale)
            # print(date)
            for i, item in enumerate(sale):
                # print(i, item)
                dt = date[i]
                nooforders = makerandomnooforders(item, customerno)
                # print(nooforders)
                
                orderdatels = [dt] * customerno
                # print(customeridls)
                # print(orderdatels)
                # print(nooforders)
                d = {'customer_id':customeridls,'order_date':orderdatels,'sale': nooforders}
                customerdata_temp = pd.DataFrame(d) 
                # print(customerdata_temp)
                customerdb = customerdb.append(customerdata_temp)
            
        
            
        # def plot():
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xlabel('Date')
            ax.set_ylabel('Sale')
            ax.set_title('Sale Trend for Customers')
            for i in range (1,customerno):
                cusid = 'C' + str(i)
                cd = customerdb.loc[customerdb['customer_id'] == cusid]
                # ax.plot(data.Date, data.Close, color='tab:blue', label='price')
                ax.plot(cd['order_date'],cd['sale'].astype(float), label = cusid)
                ax.legend(loc="upper right")
                ax.grid(True)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
            #dolwnload customer data as csv file
            csv = customerdb.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)
          
            st.pyplot(fig)
        
        

    if in_file is not None:
        os.chdir('C:/Users/FarzanehAkhbar/Documents/FAAS/data')


        st.markdown(in_file.name)
        data = pd.read_csv(in_file,  delimiter=(';'))
        #data = pd.read_csv('BTC -for_customerdatagen.csv', delimiter=(';'))
        data['date'] = pd.to_datetime(data['date'])
        data['signal'] = data['signal'].astype(float)
        dfls = [data]
        st.header("Create Customer Data")
        
        customerno = st.number_input("Enter Number of your Customers", value=1)
        st.markdown(f"Customer Number: **{customerno}**")
        if st.button('Plot '):
            plot(dfls, customerno)
        

# customerno = 20






#----------------------------------------------------------------------------------------------------


# Streamlit Demo Entry Page
def build_page_entry(session: session_state):
    st.title("Please select one of the features you want demonstrated.")


# Promotion Page
def build_page_promotion(session: session_state):
    st.title("This is the Promotions Page!")
    in_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type='csv')
    if in_file is not None:
        st.markdown(in_file.name)
        data = pd.read_csv(io.StringIO(in_file.read().decode('utf-8')), sep=',', index_col=0)
        data.index = pd.to_datetime(data.index)
        zeros = pd.DataFrame(data=range(0, len(data.index)), index=data.index) * 0
        # Marketing Promotion
        st.header("Are you going to start a promotion campaign?")
        _promo_type = st.selectbox("Promotion", ("No", "Yes"))
        if (_promo_type == "Yes"):
            c1, c2 = st.beta_columns([1, 1])
            _p1 = c1.date_input("When your promotion starts?", data.index[0])
            # _p2 = c2.date_input("When your promotion will end?",datetime.date(2020, 1, 1))
            _p2 = c2.number_input("Promotion Duration in Days", value=1)
            # start_index = random.randint(0, len(data.index)-5)
            # end_index = random.randint(start_index+1, len(data.index))
            # number_points = end_index-start_index
            # start_date = data.index[start_index]
            # date_range = pd.date_range(start=start_date, periods=number_points, freq='D')
            # promo = pd.DataFrame(data=range(0, number_points), index=date_range) * 0 + 5
            # data = data + promo
            if (_promo_type == "Yes"):
                promotion_date_range = pd.date_range(start=_p1, freq='D', periods=_p2)
                promotion = pd.DataFrame(data=range(0, _p2), index=promotion_date_range) * 0
                pro1 = promotion + .5
                for i in range(0, _p2):
                    data.loc[promotion_date_range[i]] += 1.5
        st.pyplot(make_timeseries_graph(data))



# Creates and returns graph based on Timeseries data in SessionState
def make_timeseries_graph(timeseries_data: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title('Product')
    ax.plot(timeseries_data)
    ax.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    return fig

