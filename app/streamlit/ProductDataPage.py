# Imports
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from app.streamlit.SessionState import SessionState as session_state
import app.forecasting.data_generator as gen


# Builds page for generating simple to define fake data for several products
def build_page_generate_fake_product_data(session: session_state):
    # Define Start & End Date
    col01, col02, col03 = st.beta_columns((1, 1, 1))
    in_date_start = col01.date_input("Start Date")
    in_date_end = col02.date_input("End Date")
    date_range = pd.date_range(start=in_date_start, end=in_date_end, freq='D')

    # Define Products
    in_products_text = st.text_input("List of Products. Enter products separated using commas: Product 1, Product 2, etc")
    products_list = []
    for item in in_products_text.split(","):
        item = item.strip()
        if len(item) > 0:
            products_list.append(item)

    # Build & Print Timeseries Data
    if len(products_list) > 0:
        datas = []
        graphs = []
        for item in products_list:
            data = create_fake_data(date_range)
            datas.append(data)
            graph = make_timeseries_graph(item, data)
            graphs.append(graph)
        data_total = datas[0]
        for index, data in enumerate(datas):
            if index > 0:
                data_total += data
        graph_total = make_timeseries_graph("Total Volume", data_total)
        graphs.insert(0, graph_total)
        # Print Timeseries Graphs
        split = (1, 1)
        row = 0
        while row < len(graphs)/len(split):
            cols = st.beta_columns(split)
            for loc_index, col in enumerate(cols):
                index = row*len(split) + loc_index
                if index < len(graphs):
                    col.pyplot(graphs[index])
            row += 1


# Creates some fake timeseries data and returns it in a dataframe
def create_fake_data(date_range):
    settings = gen.Settings()
    settings.set_date_range(date_range)
    settings.set_base(5)
    settings.set_trend(0.01)
    settings.add_noise_gaussian_noise(1, 0.5)
    settings.add_signal_sinusoidal(1, 0.25)
    settings.add_signal_gaussian_process(1)
    data = gen.generate(settings)[0]
    return data


# Creates and returns graph based on Timeseries data in SessionState
def make_timeseries_graph(product_name: str, timeseries_data: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title(product_name)
    ax.plot(timeseries_data)
    return fig

