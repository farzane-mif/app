# Imports
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from SessionState import SessionState as session_state
import forecasting.data_generator as gen


# Creates and returns graph based on Timeseries data
def make_timeseries_graph(timeseries_data: pd.DataFrame, title="Graph"):
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title(title)
    ax.plot(timeseries_data)
    return fig


# Creates and returns graph based on Timeseries data, rendering several columns
def make_timeseries_multi_graph(composite_data: pd.DataFrame, title="Graph"):
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title(title)
    for comp in composite_data:
        ax.plot(comp)
    return fig


# Converts provided Data Frame to csv file and constructs download link at given location
def bake_csv_data(csv_data: pd.DataFrame, link_text="Download Data", link_col=None):
    csv = csv_data.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">{link_text}</a>'
    # Create and render download link:
    write_to = st
    if link_col is not None:
        write_to = link_col
    write_to.markdown(href, unsafe_allow_html=True)

