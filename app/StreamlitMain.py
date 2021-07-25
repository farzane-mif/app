# Imports
import streamlit as st
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from SessionState import SessionState as session_state
# Import page building functions
from DataGeneratorPage import build_page_define_timeseries_data as build_page_FORECASTING_DATA
from ProductDataPage import build_page_generate_fake_product_data as build_page_PRODUCT_DATA
from Test import build_page_entry as build_page_ENTRY
from AutoGenerateDataPage import build_page_auto_generate_data
from Test import build_page_promotion as build_page_PROMOTION
from Test import build_page_test as build_page_TEST
from ForecastingPage import build_page_forecasting as build_page_FORECASTING


def init_session():
    return session_state()


# Fetch Session State
session = init_session()

# st.set_page_config(layout='wide')

# Print Heading
st.title("Timeseries Forecasting Data App")
# st.markdown(f"Session Id: **{session.session_id}**, with **{session.get_value('reloads', 0)}** reloads")
session.set_value('reloads', session.get_value('reloads', 0) + 1)

# Select featurepage to present
opt_NONE, opt_FORECAST_DATA, opt_AUTO_GENERATE, opt_PRODUCTS, opt_PROMOTIONS, opt_CUSTOMERS, opt_FORECASTING = ("None", "Forecasting Data", "Auto Generate Data", "Generate Products", "Promotions", "Customers", "Forecasting")
page_dictionary = {opt_NONE: build_page_ENTRY,
                   opt_FORECAST_DATA: build_page_FORECASTING_DATA,
                   opt_AUTO_GENERATE: build_page_auto_generate_data,
                   opt_PRODUCTS: build_page_PRODUCT_DATA,
                   opt_PROMOTIONS: build_page_PROMOTION,
                   opt_CUSTOMERS: build_page_TEST,
                   opt_FORECASTING: build_page_FORECASTING}
col01, col02 = st.beta_columns((1, 2))
col01.title("Feature:")
in_feature_selected = col02.selectbox("Select the feature you want to see:", (opt_NONE, opt_FORECAST_DATA, opt_AUTO_GENERATE, opt_PRODUCTS, opt_PROMOTIONS, opt_CUSTOMERS, opt_FORECASTING))
page_dictionary[in_feature_selected](session)

st.markdown("©2021 Mediaan, Florian Fuss & Farzaneh Akhbar")

