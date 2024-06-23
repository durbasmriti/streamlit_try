import streamlit as st

st.title("Hello, Streamlit!")
st.write("This is a simple Streamlit app.")

# Add a slider
slider_value = st.slider("Select a value", 0, 100, 50)
st.write("Selected value:", slider_value)

# Add a button
if st.button("Click Me"):
    st.write("Button clicked!")
