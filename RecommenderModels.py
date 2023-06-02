from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['article_id'].isin(items_to_ignore)] \
                                                .sort_values('popularity', ascending=False)\
                                                .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'article_id', 
                                                          right_on = 'article_id')[['popularity', 'article_id']]


        return recommendations_df
    

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['article_id'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'article_id', 
                                                          right_on = 'article_id')[['recStrength', 'article_id']]


        return recommendations_df
    

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, clicks, items_df):
        self.clicks = clicks
        self.items_df = items_df[['category_id', 'created_at_ts', 'words_count']]
        self.embeddings_df =  items_df.drop(columns=['category_id', 'created_at_ts', 'words_count'])

        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def get_user_profile(self, user_id):
        user_read_articles_id = list(set(self.clicks[self.clicks.user_id == user_id].article_id))
        user_profile = self.embeddings_df.loc[user_read_articles_id]
        return user_profile
    
    def recommend_items(self, user_id, topn=5):
        user_profile = self.get_user_profile(user_id)
        cosine_similarities = cosine_similarity(user_profile, self.embeddings_df)
        max_similarities = cosine_similarities.max(axis=0)
        articles_similarities = pd.DataFrame({'article_id': self.embeddings_df.index, 'max_similarity': max_similarities})
        # Filtrer les articles déjà lus
        articles_similarities = articles_similarities[~articles_similarities['article_id'].isin(list(user_profile.index))]
        # Trier le DataFrame par ordre décroissant de similarité
        articles_similarities = articles_similarities.sort_values('max_similarity', ascending=False)
        return articles_similarities.head(topn)
