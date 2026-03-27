import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

st.title("GATEWAYS-2025-National Level Fest Data Analysis")

df = pd.read_csv("C5-FestDataset - fest_dataset.csv")

# dataset columns - Student Name, College, Phone Number, Place, State, Event Name, Event Type, Amount Paid, Feedback on Fest, Rating

st.subheader("Dataset")

if st.button("Click on me to View the Dataset"):
    with st.expander("Dataset Preview"):
        st.dataframe(df)

st.markdown("---")

# 1 - analysis of particpants trends (event-wise, college-wise, etc.); plot the statewise participants in India map

# analysis of participants trends

st.subheader("Participants Trends")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Participants", len(df))
with col2:
    st.metric("Total Colleges", len(df["College"]))
with col3:
    st.metric("Total Events", len(df["Event Name"]))
with col4:
    st.metric("Event Types", len(df["Event Type"]))
with col5:
    st.metric("Total Amount Paid", "\u20B9" + str(df["Amount Paid"].sum()))

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Participants Trends", "Participants Text Feedback", "Participants Ratings", "Interactive Dashboard"])

# plot the statewise participants in India map

with tab1:

    st.subheader("Statewise Participants in India")

    # calculate the frequency
    state_counts = df.groupby('State')['Student Name'].count().reset_index()
    state_counts.columns = ['State', 'participant_count']

    # load the shapefile
    india_map = gpd.read_file('in_shp/in.shp')
    # columns - id, name, source, geometry

    # merge shapefile with participant counts
    merged_map = india_map.merge(state_counts, left_on='name', right_on='State', how='left')

    # fill with 0 for states that have no participants
    merged_map['participant_count'] = merged_map['participant_count'].fillna(0)

    # plot the map
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # plotting with color frequency
    merged_map.plot(
        column='participant_count',
        cmap='Blues',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': "Number of Participants by State"}
    )
    plt.title('Statewise Participants in India', fontdict={'fontsize': 16})
    plt.axis('off')
    st.pyplot(fig)

# 2 - analysis of participants text feedback and ratings

# participants text feedback
with tab2:
    st.subheader("Participants Text Feedback")

    with st.expander("View Feedback"):
        st.dataframe(df["Feedback on Fest"])
    
    st.text("Total Feedback Received: " + str(len(df["Feedback on Fest"])) + " Feedbacks")

    st.text("Text Cleaning")

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split() if w not in stop_words]
        return " ".join(words)

    # apply the function to the feedback column
    df["Cleaned Feedback"] = df["Feedback on Fest"].apply(preprocess)

    with st.expander("View Cleaned Data"):
        st.dataframe(df["Cleaned Feedback"])

    # combine all the feedback text into 1 single string
    feedback_text = " ".join(df["Cleaned Feedback"].dropna().astype(str))

    # generate wordcloud
    wordcloud = WordCloud(
        width = 800,
        height = 400,
        background_color = 'white',
        max_words = 300,
        colormap = 'viridis'
    ).generate(feedback_text)

    # plot the wordcloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Feedback WordCloud", fontsize=16)
    st.pyplot(fig)

# participants ratings
with tab3:
    st.subheader("Participants Ratings")

    with st.expander("View Ratings"):
        st.dataframe(df["Rating"])

    st.text("Total Ratings Received: " + str(len(df["Rating"])) + " Ratings")

    # clean the preprocess the ratings column
    cleaned_ratings = pd.to_numeric(df["Rating"], errors="coerce").dropna()

    st.text("total valid ratings: " + str(len(cleaned_ratings)) + " Ratings")

    if len(cleaned_ratings) > 0:
        # show average
        average_rating = cleaned_ratings.mean()
        st.metric(label = "Average Rating", value = f"{average_rating:.2f}")

        # show min and max
        min_rating = cleaned_ratings.min()
        max_rating = cleaned_ratings.max()
        st.metric(label = "Minimum Rating", value = f"{min_rating:.2f}")
        st.metric(label = "Maximum Rating", value = f"{max_rating:.2f}")

        # bar chart for rating distribution

        st.text("Rating Distribution")

        with st.expander("View Rating Distribution Chart", expanded = True):
            # count the frequency of each ratings
            rating_counts = cleaned_ratings.value_counts().sort_index()

            # plot
            fig, ax = plt.subplots(figsize=(8, 4))
            rating_counts.plot(kind='bar', ax=ax)
            ax.set_title("How many participants gave each rating?")
            ax.set_xlabel("Rating Value")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)

# 3 - interactive dashboard
with tab4:
    st.subheader("Interactive Dashboard")

    st.markdown("dashboard filters")

    all_states = df['State'].dropna().unique()
    all_events = df['Event Type'].dropna().unique()

    filt_col1, filt_col2 = st.columns(2)
    
    with filt_col1:
        selected_states = st.multiselect("Filter by State:", all_states, default=all_states)
        
    with filt_col2:
        selected_events = st.multiselect("Filter by Event Type:", all_events, default=all_events)
    
    # apply the filter
    dash_df = df[
        (df['State'].isin(selected_states)) & 
        (df['Event Type'].isin(selected_events))
    ]
    
    st.markdown("---")
    # columns for the first row of charts
    col1, col2 = st.columns(2)
    
    # pie chart
    with col1:
        st.text("Event Type Breakdown")
        event_counts = dash_df['Event Type'].value_counts()
        
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        ax_pie.pie(
            event_counts, 
            labels=event_counts.index, 
            autopct='%1.1f%%', 
            startangle=140,
            colors=plt.cm.Set3.colors,
            wedgeprops={'edgecolor': 'white'}
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
        
    # bar chart for average rating
    with col2:
        st.text("Average Rating by Event Type")
        
        # clean the rating data for calculation
        dash_df_clean = dash_df.copy()
        dash_df_clean['Rating'] = pd.to_numeric(dash_df_clean['Rating'], errors='coerce')
        
        # group by event type to get the average rating
        avg_rating = dash_df_clean.groupby('Event Type')['Rating'].mean().dropna()
        
        fig_bar, ax_bar = plt.subplots(figsize=(5, 5))
        
        # bar chart
        avg_rating.plot(kind='bar', ax=ax_bar, edgecolor='black')
        
        ax_bar.set_xlabel("Event Type")
        ax_bar.set_ylabel("Average Rating (Out of 5)")
        ax_bar.set_ylim(0, 5.5)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.xticks(rotation=45, ha='right') # rotate labels to avoid overlap
        st.pyplot(fig_bar)

    
st.markdown("---")