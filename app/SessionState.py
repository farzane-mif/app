import streamlit as st
import random


# Private dictionary that stores a dictionary for each active session, with keys corresponding to session_id
_sessions_dictionary = {}


# Returns value of given key for current session. Returns default if key not defined (Does not set key or key value)
def get_value(session_id, key, default=None):
    session = _sessions_dictionary.get(session_id, {})
    out = session.get(key, default)
    return out


# Sets value of given key for current session
def set_value(session_id, key, value):
    session = _sessions_dictionary.get(session_id, {})
    session[key] = value
    _sessions_dictionary[session_id] = session


# Utility that returns session id from URL or if none present: creates, sets to URL and returns new id
def init_session_id():
    try:
        app_state = st.experimental_get_query_params()
        session_id = int(app_state['session'][0])
    except:
        session_id = random.randint(0, 9999999999999999)
        st.experimental_set_query_params(session=session_id)
    return session_id


# Utility Class that handles all session id tasks and delegates get and set value functions
class SessionState:

    # Init SessionState and fetch/create session id
    def __init__(self):
        self.session_id = init_session_id()

    # Delegates to get_value using this class's session_id
    def get_value(self, key, default=None):
        return get_value(self.session_id, key, default)

    # Delegates to set_value using this class's session_id
    def set_value(self, key, value):
        set_value(self.session_id, key, value)

