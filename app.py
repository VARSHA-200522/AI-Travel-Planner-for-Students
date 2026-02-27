import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ----------------------------
# Sample Travel Dataset
# ----------------------------

data = {
    "Destination": ["Goa", "Manali", "Jaipur", "Ooty", "Kerala", "Pondicherry", "Darjeeling"],
    "Type": ["Beach", "Hill", "Heritage", "Hill", "Nature", "Beach", "Hill"],
    "Avg_Cost": [8000, 7000, 6000, 6500, 9000, 5500, 7500],
    "Rating": [4.5, 4.6, 4.2, 4.3, 4.8, 4.1, 4.4]
}

df = pd.DataFrame(data)

# ----------------------------
# K-Means Clustering
# ----------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Avg_Cost", "Rating"]])

# ----------------------------
# Cost Prediction Model
# ----------------------------

X = df[["Rating"]]
y = df["Avg_Cost"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🎒 AI Travel Planner for Students")

st.sidebar.header("Enter Your Preferences")

budget = st.sidebar.number_input("Enter your budget (₹)", min_value=3000, max_value=20000, step=1000)
interest = st.sidebar.selectbox("Select your interest", df["Type"].unique())
days = st.sidebar.slider("Number of travel days", 1, 7, 3)

# ----------------------------
# Recommendation Logic
# ----------------------------

filtered = df[df["Type"] == interest]
recommended = filtered[filtered["Avg_Cost"] <= budget]

if st.button("Get Travel Plan"):

    if recommended.empty:
        st.warning("No destinations found within your budget.")
    else:
        destination = recommended.iloc[0]

        st.success(f"Recommended Destination: {destination['Destination']}")

        predicted_cost = model.predict([[destination["Rating"]]])[0]

        st.write(f"Estimated Trip Cost: ₹{int(predicted_cost)}")

        st.subheader("🗓 Day-wise Itinerary")

        for day in range(1, days + 1):
            st.write(f"Day {day}: Explore top attractions in {destination['Destination']}")

        st.subheader("📊 Destination Cluster Info")
        st.write(f"Cluster Group: {destination['Cluster']}")

        st.balloons()
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ----------------------------
# Sample Travel Dataset
# ----------------------------

data = {
    "Destination": ["Goa", "Manali", "Jaipur", "Ooty", "Kerala", "Pondicherry", "Darjeeling"],
    "Type": ["Beach", "Hill", "Heritage", "Hill", "Nature", "Beach", "Hill"],
    "Avg_Cost": [8000, 7000, 6000, 6500, 9000, 5500, 7500],
    "Rating": [4.5, 4.6, 4.2, 4.3, 4.8, 4.1, 4.4]
}

df = pd.DataFrame(data)

# ----------------------------
# K-Means Clustering
# ----------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Avg_Cost", "Rating"]])

# ----------------------------
# Cost Prediction Model
# ----------------------------

X = df[["Rating"]]
y = df["Avg_Cost"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🎒 AI Travel Planner for Students")

st.sidebar.header("Enter Your Preferences")

budget = st.sidebar.number_input(
    "Enter your budget (₹)",
    min_value=3000,
    max_value=20000,
    step=1000,
    key="budget_input"
)

interest = st.sidebar.selectbox(
    "Select your interest",
    df["Type"].unique(),
    key="interest_select"
)

days = st.sidebar.slider(
    "Number of travel days",
    1,
    7,
    3,
    key="days_slider"
)

# ----------------------------
# Recommendation Logic
# ----------------------------

filtered = df[df["Type"] == interest]
recommended = filtered[filtered["Avg_Cost"] <= budget]

if st.button("Get Travel Plan", key="travel_button"):

    if recommended.empty:
        st.warning("No destinations found within your budget.")
    else:
        destination = recommended.iloc[0]

        st.success(f"Recommended Destination: {destination['Destination']}")

        predicted_cost = model.predict([[destination["Rating"]]])[0]

        st.write(f"Estimated Trip Cost: ₹{int(predicted_cost)}")

        st.subheader("🗓 Day-wise Itinerary")

        for day in range(1, days + 1):
            st.write(f"Day {day}: Explore top attractions in {destination['Destination']}")

        st.subheader("📊 Destination Cluster Info")
        st.write(f"Cluster Group: {destination['Cluster']}")

        st.balloons()