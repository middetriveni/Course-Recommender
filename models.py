import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_5_models_recommendations(df, course_name):
    # 1. Popularity-Based Model (High Ratings)
    pop_df = df.sort_values(by=['rating', 'enrollment_numbers'], ascending=False).head(5)
    
    # 2. Content-Based Model (Similar Courses)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['course_name'])
    sig = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    try:
        idx = df.index[df['course_name'] == course_name][0]
        sig_scores = list(enumerate(sig[idx]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)[1:6]
        content_indices = [i[0] for i in sig_scores]
        content_df = df.iloc[content_indices]
    except:
        content_df = df.head(5)

    # 3. Same Difficulty Model
    diff_val = df[df['course_name'] == course_name]['difficulty_level'].values[0]
    diff_df = df[df['difficulty_level'] == diff_val].sample(5)
    
    # 4. Budget-Friendly Model (Cheapest)
    cheap_df = df.sort_values(by='course_price', ascending=True).head(5)
    
    # 5. Top Instructor Model (Feedback Score)
    inst_df = df.sort_values(by='feedback_score', ascending=False).head(5)
    
    return pop_df, content_df, diff_df, cheap_df, inst_df
