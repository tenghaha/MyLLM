import streamlit as st


pages = {
    "导航": [
        st.Page("prompts.py", title="Prompts", icon=":material/sms:"),
        st.Page("assistant.py", title="Assistant", icon=":material/smart_toy:")
    ]
}

pg = st.navigation(pages)
pg.run()