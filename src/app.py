import streamlit as st
from model import predict

st.set_page_config(page_title="HS Detection", page_icon=":warning:", layout="centered")

st.title("Speech detection in Social media")
st.write("you can paste your text in the  feilf below to check if it contains any vulgar meanings")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("check"):
    if not user_input.strip():
        st.warning("Please enter or paste text")
    else:
        result = predict(user_input)
        if result == "Hate Speech":
            st.error(f"Result: **{result}**")
        elif result == "Offensive Language":
            st.warning(f"Result: **{result}**")
        else:
            st.success(f"Result: **{result}**")