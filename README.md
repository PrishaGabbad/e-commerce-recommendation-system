PROJECT OVERVIEW – Flipkart Product Recommendation System
This project builds a hybrid product recommendation system using a sample Flipkart e-commerce dataset. The aim is to recommend relevant products to users using multiple approaches such as rating-based filtering, content-based filtering, and collaborative filtering.
The project also includes data cleaning, visualization, NLP preprocessing with spaCy, TF-IDF vectorization, user similarity calculations, and a complete prototype of a recommender engine.
Project Features
Data Cleaning & Pre-processing
Handling missing values
Converting text fields to usable forms
Preparing numeric columns
Extracting important keywords using spaCy NLP
Exploratory Data Analysis & Visualization
Heatmap of average ratings by brands
Distribution chart of overall ratings
Identification of most popular products
NLP-Based Tag Extraction
Cleaning category, brand, and description text
Removing stopwords
Extracting important tokens
Creating a combined Tags field for content-based recommendations
Rating-Based Recommendation System
Recommends the top trending/highest-rated products by:
Grouping by product
Calculating average ratings
Sorting in descending order
Content-Based Recommendation System (TF-IDF + Cosine Similarity)
Recommends products similar to a chosen product based on:
Tags generated from product category, brand, and description
TF-IDF vectorization
Cosine similarity score
Example: If the user views “Ladela Bellies”, the system recommends similar shoes.
Collaborative Filtering Recommendation System
Simulates user behavior to generate:
User–Item Matrix
User–User similarity matrix
Item suggestions based on similar users
This mimics real-world recommender systems used by e-commerce platforms.
Major Features Summary
Feature	Description
Data Cleaning	Replacing nulls, formatting categorical & numeric fields
NLP Tag Extraction	spaCy tokenization + stopword removal
TF-IDF Model	Vectorizing product descriptions/tags
Content-Based Filtering	Similarity between product vectors
Collaborative Filtering	Similar users → similar items
Data Visualization	Heatmaps & bar charts using Matplotlib & Seaborn
Technology / Tools Used
Programming Language
Python 3
Libraries / Frameworks
Category	Tools
Data Processing	pandas, numpy
Visualization	matplotlib, seaborn
Machine Learning	scikit-learn (TF-IDF, cosine similarity)
NLP	spaCy (en_core_web_sm)
Math/Matrix Ops	scipy
Environment	Google Colab / Jupyter Notebook
Data
Flipkart E-commerce Sample Dataset (CSV)
Software Used
Python 3.x
Jupyter Notebook / Google Colab
spaCy
scikit-learn
pandas, numpy, scipy
Matplotlib & Seaborn
Steps to Install & Run the Project
Step 1: Install Dependencies
Open Colab or your local environment and install:
pip install pandas numpy matplotlib seaborn scikit-learn spacy
python -m spacy download en_core_web_sm
Step 2: Upload Dataset
In Google Colab:
from google.colab import files
uploaded = files.upload()
Upload:
flipkart_com-ecommerce_sample.csv
Step 3: Load and Clean Data
train_data = pd.read_csv('flipkart_com-ecommerce_sample.csv')
Run all preprocessing steps:
selecting columns
filling null values
cleaning text
extracting tags
Step 4: Generate Visualizations
Run the sections generating:
Heatmap of brand ratings
Popular products chart
Ratings distribution chart
Step 5: Build Recommendation Systems
Rating-Based:
rating_base_recommendation.head(10)
Content-Based:
recommendations = content_based_recommendations(train_data, 'Ladela Bellies')
Collaborative Filtering:
collab_rec = collaborative_filtering_recommendations(train_data, 4, 5)
Instructions for Testing the Project
Test 1: View Missing Data
train_data.isnull().sum()
Test 2: Check Tag Extraction
train_data[['product_name','Tags']].head()
Test 3: Test Content-Based Recommendation
Try different products:
content_based_recommendations(train_data, 'Apple iPhone 6', 5)
Test 4: Test Collaborative Filtering
Change user ID and top_n:
collaborative_filtering_recommendations(train_data, 10, 8)
Test 5: Check Visualizations
All plots should appear correctly:
Heatmap
Popular products bar chart
Ratings distribution
