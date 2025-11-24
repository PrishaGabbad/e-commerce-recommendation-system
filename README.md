PROJECT OVERVIEW – <br> Flipkart Product Recommendation System <br>

This project builds a hybrid product recommendation system using a sample Flipkart e-commerce dataset. The aim is to recommend relevant products to users using multiple approaches such as rating-based filtering, content-based filtering, and collaborative filtering. <br>

The project also includes data cleaning, visualization, NLP preprocessing with spaCy, TF-IDF vectorization, user similarity calculations, and a complete prototype of a recommender engine. <br>

Project Features <br>
Data Cleaning & Pre-processing
Handling missing values
Converting text fields to usable forms
Preparing numeric columns
Extracting important keywords using spaCy NLP <br>

Exploratory Data Analysis & Visualization <br>
Heatmap of average ratings by brands
Distribution chart of overall ratings
Identification of most popular products <br>

NLP-Based Tag Extraction <br>
Cleaning category, brand, and description text
Removing stopwords
Extracting important tokens
Creating a combined Tags field for content-based recommendations <br>

Rating-Based Recommendation System <br>
Recommends the top trending/highest-rated products by:
Grouping by product
Calculating average ratings
Sorting in descending order <br>

Content-Based Recommendation System (TF-IDF + Cosine Similarity) <br>
Recommends products similar to a chosen product based on:
Tags generated from product category, brand, and description
TF-IDF vectorization
Cosine similarity score
Example: If the user views “Ladela Bellies”, the system recommends similar shoes. <br>

Collaborative Filtering Recommendation System <br>
Simulates user behavior to generate:
User–Item Matrix
User–User similarity matrix
Item suggestions based on similar users
This mimics real-world recommender systems used by e-commerce platforms. <br>

Major Features Summary <br>
Feature	Description
Data Cleaning	Replacing nulls, formatting categorical & numeric fields
NLP Tag Extraction	spaCy tokenization + stopword removal
TF-IDF Model	Vectorizing product descriptions/tags
Content-Based Filtering	Similarity between product vectors
Collaborative Filtering	Similar users → similar items
Data Visualization	Heatmaps & bar charts using Matplotlib & Seaborn <br>

Technology / Tools Used <br>
Programming Language <br>
Python 3 <br>
Libraries / Frameworks <br>
Category	Tools  <br>
Data Processing	pandas, numpy
Visualization	matplotlib, seaborn
Machine Learning	scikit-learn (TF-IDF, cosine similarity)
NLP	spaCy (en_core_web_sm)
Math/Matrix Ops	scipy
Environment	Google Colab / Jupyter Notebook <br>
Data <br>
Flipkart E-commerce Sample Dataset (CSV) <br>

Software Used <br>
Python 3.x <br>
Jupyter Notebook / Google Colab <br>
spaCy <br>
scikit-learn <br>
pandas, numpy, scipy <br>
Matplotlib & Seaborn <br> <br>
 
Steps to Install & Run the Project <br>
Step 1: Install Dependencies <br>
Open Colab or your local environment and install: <br>
pip install pandas numpy matplotlib seaborn scikit-learn spacy
python -m spacy download en_core_web_sm <br> <br>

Step 2: Upload Dataset <br>
In Google Colab: <br>
from google.colab import files 
uploaded = files.upload() <br>
Upload: <br>
flipkart_com-ecommerce_sample.csv <br> <br>

Step 3: Load and Clean Data <br>
train_data = pd.read_csv('flipkart_com-ecommerce_sample.csv') <br>
Run all preprocessing steps: <br>
selecting columns <br>
filling null values <br>
cleaning text <br>
extracting tags <br> <br>

Step 4: Generate Visualizations <br>
Run the sections generating: <br>
Heatmap of brand ratings <br>
Popular products chart <br>
Ratings distribution chart <br> <br> 

Step 5: Build Recommendation Systems
Rating-Based: <br>
rating_base_recommendation.head(10)
Content-Based: <br>
recommendations = content_based_recommendations(train_data, 'Ladela Bellies') 
Collaborative Filtering:
collab_rec = collaborative_filtering_recommendations(train_data, 4, 5) <br> <br>
Instructions for Testing the Project <br>
Test 1: View Missing Data <br>
train_data.isnull().sum() <br> <br>
Test 2: Check Tag Extraction <br>
train_data[['product_name','Tags']].head() <br> <br>
Test 3: Test Content-Based Recommendation  <br>
Try different products: <br>
content_based_recommendations(train_data, 'Apple iPhone 6', 5) <br> <br>
Test 4: Test Collaborative Filtering <br>
Change user ID and top_n: <br>
collaborative_filtering_recommendations(train_data, 10, 8) <br> <br>
Test 5: Check Visualizations <br>
All plots should appear correctly: <br>
Heatmap <br> 
Popular products bar chart <br>
Ratings distribution <br>
