import streamlit as st

st.set_page_config(page_title="MyLLM", layout="wide")

pages = {
    "导航": [
        st.Page("prompts.py", title="Prompts", icon=":material/sms:"),
        st.Page("assistant.py", title="Assistant", icon=":material/smart_toy:")
    ]
}

pg = st.navigation(pages)
pg.run()