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
import Util as util


# Keys used in SessionState
KEY_TIMESERIES_DATA = 'timeseries_data'
KEY_TIMESERIES_COMPOSITE = 'timeseries_composite_data'
KEY_TIMESERIES_DATA_GRAPH = 'timeseries_data_graph'
KEY_TIMESERIES_COMPOSITE_GRAPH = 'timeseries_composite_data_graph'


# Main method constructing the page (Streamlit module, SessionState object)
def build_page_define_timeseries_data(session: session_state):
    # Define Settings object for Data Generation
    gen_settings = gen.Settings()

    # Time Settings
    opt_DAILY, opt_WEEKLY, opt_MONTHLY, opt_QUATERLY, opt_YEARLY = ("Daily", "Weekly", "Monthly", "Quaterly", "Yearly")
    col01, col02, col03 = st.beta_columns((1, 1, 1))
    date_start = col01.date_input('Start Date')
    date_number_entries = col02.number_input("Number of Entries", value=100, step=1, min_value=1)
    date_interval = col03.selectbox("Interval between Entries", (opt_DAILY, opt_WEEKLY, opt_MONTHLY, opt_QUATERLY, opt_YEARLY))
    date_range = gen.util_make_daterange(date_start, date_number_entries)
    st.markdown(f"From **{date_range[0].date()}** to **{date_range[-1].date()}**")
    gen_settings.set_date_range(date_range)

    # Base Volume & Trend
    col01, col02, col03 = st.beta_columns((1, 1, 1))
    gen_base = col01.number_input("Base Volume", value=0.0, step=0.1)
    gen_trend = col02.number_input("Trend (per 100 Entries)", value=0.0, step=0.1)
    gen_total_impact = col03.number_input("Total Impact", value=1.0, step=0.1)
    gen_settings.set_base(gen_base)
    gen_settings.set_trend(gen_trend/100)
    gen_settings.set_total_impact(gen_total_impact)

    # Noise
    opt_NONE, opt_WHITE, opt_RED = ["None", "White", "Red"]
    col01, col02, col03, col04 = st.beta_columns((1, 1, 1, 1))
    noise_type = col01.selectbox("Noise Type", (opt_NONE, opt_WHITE, opt_RED))
    if noise_type != opt_NONE:
        noise_impact = col02.number_input("Impact", value=1.0, step=0.05, min_value=0.0)
        noise_std = col03.number_input("Std", value=0.1, step=0.05, min_value=0.0)
    if noise_type == opt_WHITE:
        gen_settings.add_noise_gaussian_noise(impact=noise_impact, std=noise_std)
    elif noise_type == opt_RED:
        noise_tau = col04.number_input("Tau", value=0.8, step=0.05)
        gen_settings.add_noise_red_noise(impact=noise_impact, std=noise_std, tau=noise_tau)

    # Signal TODO: Add Sampling options
    opt_NONE, opt_SIN, opt_GAUSS, opt_AR, opt_CAR, opt_PP, opt_MG, opt_NARMA = ("None", "Sinusoidal", "Gaussian", "Auto Regressive", "CAR", "Pseudo Periodic", "Mackey Glass", "NARMA")
    signals_number = st.number_input("Number of Signals", value=0, step=1, min_value=0)
    for sc in range(0, signals_number):
        col01, col02, col03, col04 = st.beta_columns((1, 1, 1, 1))
        signal_type = col01.selectbox("Signal Type", (opt_NONE, opt_SIN, opt_GAUSS, opt_AR, opt_CAR, opt_MG, opt_NARMA), key=f'signal_{sc}_type')
        if signal_type != opt_NONE:
            signal_impact = col02.number_input("Impact", value=1.0, step=0.05, min_value=0.0, key=f'signal_{sc}_impact')
        if signal_type == opt_SIN:
            signal_par01 = col03.number_input("Frequency", value=0.25, step=0.05, min_value=0.0, key=f'signal_{sc}_par01')
            gen_settings.add_signal_sinusoidal(signal_impact, frequency=signal_par01)
        elif signal_type == opt_GAUSS:
            opt_Kernel_MATERN, _ = ["Matern", None]  # TODO: Allow for other kernals {'SE', 'Constant', 'Exponential', 'RQ', 'Linear', 'Matern', 'Periodic'}
            signal_par01 = col03.selectbox("Signal Type", [opt_Kernel_MATERN], key=f'signal_{sc}_par01')
            signal_par02 = col04.number_input("nu", value=3. / 2, step=0.1, min_value=0.0, key=f'signal_{sc}_par02')
            gen_settings.add_signal_gaussian_process(signal_impact, kernel=signal_par01, nu=signal_par02)
        elif signal_type == opt_AR:
            signal_par01 = col03.number_input("AR Parameter 1", value=1.5, step=0.05, key=f'signal_{sc}_par01')
            signal_par02 = col04.number_input("AR Parameter 2", value=-0.75, step=0.05, key=f'signal_{sc}_par02')
            gen_settings.add_signal_auto_regressive(signal_impact, ar_param=[signal_par01, signal_par02])
        elif signal_type == opt_CAR:
            signal_par01 = col03.number_input("AR Parameter", value=0.9, step=0.05, key=f'signal_{sc}_par01')
            signal_par02 = col04.number_input("Sigma", value=-0.2, step=0.05, key=f'signal_{sc}_par02')
            gen_settings.add_signal_car(signal_impact, ar_param=signal_par01, sigma=signal_par02)
        elif signal_type == opt_NARMA:
            signal_par01 = col03.number_input("Order", value=10, step=1, min_value=1, key=f'signal_{sc}_par01')
            gen_settings.add_signal_narma(signal_impact, order=signal_par01)

    # Update Graph Button
    col01, col02 = st.beta_columns((2, 1))
    is_update = col01.button('Update Data & Graph')
    if is_update or session.get_value(KEY_TIMESERIES_DATA, None) is None:
        update_timeseries_data(session, gen_settings)
        util.bake_csv_data(session.get_value(KEY_TIMESERIES_DATA), link_text="Download Timeseries Data", link_col=col02)

    # Graphs
    col01, col02 = st.beta_columns([1, 1])
    col01.pyplot(session.get_value(KEY_TIMESERIES_DATA_GRAPH))
    col02.pyplot(session.get_value(KEY_TIMESERIES_COMPOSITE_GRAPH))

    # Debug Code Printout
    st.code(gen.util_stringify_settings(gen_settings), language="python")


# Updates Timeseries Data & Composite
def update_timeseries_data(session: session_state, settings: gen.Settings):
    data, composite = gen.generate(settings)
    session.set_value(KEY_TIMESERIES_DATA, data)
    session.set_value(KEY_TIMESERIES_COMPOSITE, composite)
    graph = util.make_timeseries_graph(session.get_value(KEY_TIMESERIES_DATA), "Autogenerated Timeseries Data")
    session.set_value(KEY_TIMESERIES_DATA_GRAPH, graph)
    comp_graph = util.make_timeseries_multi_graph(session.get_value(KEY_TIMESERIES_COMPOSITE), "Autogenerated Timeseries Components")
    session.set_value(KEY_TIMESERIES_COMPOSITE_GRAPH, comp_graph)

