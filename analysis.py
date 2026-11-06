import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: DATASET READING

df = pd.read_csv("SpotifyDataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# STEP 2: DATA UNDERSTANDING

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nMissing Values:")
print(df.isnull().sum())


# STEP 3: DATA TRANSFORMATION

# Convert duration from seconds to minutes
df['Duration_min'] = df['Duration'] / 60

# Convert Year to integer
df['Year'] = df['Year'].astype(int)

# Remove duplicate records
df = df.drop_duplicates()

print("\nTransformed Dataset:")
print(df.head())

print("\nUpdated Dataset Shape:")
print(df.shape)


# STEP 4: HANDLING NULL VALUES

print("\nMissing Values Before Imputation:")
print(df.isnull().sum())

# Mean imputation for numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Mode imputation for categorical columns
text_columns = df.select_dtypes(include=['object', 'str']).columns

for col in text_columns:
    if df[col].isnull().any() and not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())


# STEP 5: INSIGHT GENERATION

# 1. Top 10 songs by streams
print("\nTop 10 Songs by Streams:")
print(df.nlargest(10, 'Streams')[['Song', 'Artist', 'Streams']])

# 2. Top 10 artists by total streams
print("\nTop 10 Artists:")
print(df.groupby('Artist')['Streams'].sum().sort_values(ascending=False).head(10))

# 3. Average popularity
print("\nAverage Popularity:")
print(df['Popularity'].mean())

# 4. Average song duration
print("\nAverage Song Duration (minutes):")
print(df['Duration_min'].mean())

# 5. Top 10 genres by average popularity
print("\nTop 10 Popular Genres:")
print(df.groupby('Genre')['Popularity'].mean().sort_values(ascending=False).head(10))

# 6. Average danceability
print("\nAverage Danceability:")
print(df['Danceability'].mean())

# 7. Average energy
print("\nAverage Energy:")
print(df['Energy'].mean())

# 8. Year-wise total streams
print("\nYear-wise Total Streams:")
print(df.groupby('Year')['Streams'].sum())


# STEP 6: DATA VISUALIZATION

# 1. Bar Chart - Top 10 Songs by Streams

top_songs = df.nlargest(10, 'Streams')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_songs, x='Streams', y='Song')
plt.title("Top 10 Songs by Spotify Streams")
plt.xlabel("Total Streams")
plt.ylabel("Song Name")
plt.tight_layout()
plt.savefig("chart1.png")
plt.show()
plt.close()


# 2. Line Chart - Year-wise Total Streams

yearly_streams = df.groupby('Year')['Streams'].sum()

plt.figure(figsize=(10, 5))
yearly_streams.plot(kind='line', marker='o')
plt.title("Year-wise Total Spotify Streams")
plt.xlabel("Year")
plt.ylabel("Total Streams")
plt.grid(True)
plt.tight_layout()
plt.savefig("chart2.png")
plt.show()
plt.close()


# 3. Scatter Plot - Popularity vs Streams

plt.figure(figsize=(8, 5))
plt.scatter(df['Streams'], df['Popularity'], alpha=0.5)
plt.title("Popularity vs Spotify Streams")
plt.xlabel("Total Streams")
plt.ylabel("Popularity")
plt.tight_layout()
plt.savefig("chart3.png")
plt.show()
plt.close()


# 4. Bar Chart - Top 10 Popular Genres

top_genres = df.groupby('Genre')['Popularity'].mean().nlargest(10)

plt.figure(figsize=(10, 5))
top_genres.plot(kind='bar')
plt.title("Top 10 Popular Music Genres")
plt.xlabel("Genre")
plt.ylabel("Average Popularity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chart4.png")
plt.show()
plt.close()


# SAVE CLEANED DATASET

df.to_csv("Spotify_Cleaned.csv", index=False)

print("\nAll 4 charts saved successfully!")
print("Cleaned dataset saved successfully!")