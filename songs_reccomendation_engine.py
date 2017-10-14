import pandas as pd
from sklearn.model_selection import train_test_split
import Recommender as rd


triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

#song_df_1 = pd.read_table("data/10000.txt",header=None)
song_df_1 = pd.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#song_df_2 =  pd.read_csv("data/song_data.csv")
song_df_2 =  pd.read_csv(songs_metadata_file)
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

print song_df.head()

song_grouped = song_df.groupby(['song_id']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song_id'], ascending = [0,1])


users = song_df['user_id'].unique()
print "Number of Users: ",len(users) 

songs = song_df['song_id'].unique()
print "Number of Songs: ",len(songs)

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)


# Recommend by Popularity
pm = rd.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song_id')

#user the popularity model to make some prediction
user_id = users[5]
recomm = pm.recommend(user_id)


print "\n--------------------Recommend by Popularity------------------"
print recomm


#Collaborative Based
#Item-item filtering

is_model = rd.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song_id')

user_id = users[5]
user_items = is_model.get_user_items(user_id)


print "\n--------------------Recommend by Item-Item Collaborative Filtering------------------"

print("\nTraining data songs for the user userid: %s:" % user_id)

for user_item in user_items:
    print user_item

#Recommend songs for the user using personalized model
print "Recommended Sonds:" ,is_model.recommend(user_id)
