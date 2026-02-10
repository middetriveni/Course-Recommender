import streamlit as st
import pandas as pd
from models import get_5_models_recommendations

st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")
st.title("🎓 Smart Course Recommendation System")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv", on_bad_lines='skip', engine='python')

df = load_data()


st.sidebar.header("Course Selection")
course_list = df['course_name'].unique()
selected_course = st.sidebar.selectbox("Edo oka course select cheyandi:", course_list)

if st.sidebar.button('Show Recommendations'):
   
    m1, m2, m3, m4, m5 = get_5_models_recommendations(df, selected_course)
    
    st.success(f"Selected: {selected_course}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Popular", "Similar", "Difficulty", "Budget", "Instructors"])

    with tab1: st.table(m1[['course_name', 'rating']])
    with tab2: st.table(m2[['course_name', 'difficulty_level']])
    with tab3: st.table(m3[['course_name', 'instructor']])
    with tab4: st.table(m4[['course_name', 'course_price']])
    with tab5: st.table(m5[['course_name', 'feedback_score']])
