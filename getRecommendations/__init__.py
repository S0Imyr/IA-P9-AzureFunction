
import logging
import json

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import azure.functions as func

import RecommenderModels 

def main(req: func.HttpRequest, interactions: func.InputStream, articles: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    # Configuration des en-têtes CORS
    headers = {
        "Access-Control-Allow-Origin": "https://ia-projet9-webapp.azurewebsites.net",  
        "Access-Control-Allow-Methods": "GET, OPTIONS", 
        "Access-Control-Allow-Headers": "Content-Type"  
    }

    if req.method == "OPTIONS":
        # Réponse préalable pour les requêtes OPTIONS
        return func.HttpResponse(headers=headers)

    # Lecture des fichiers CSV des interactions etv articles dans un DataFrame
    interactions_df = pd.read_csv(interactions)
    articles_df = pd.read_csv(articles, index_col='article_id')


    # Work for Popularity model
    popularity_df = pd.DataFrame({
        'article_id': interactions_df['article_id'].value_counts().index,
        'popularity': interactions_df['article_id'].value_counts().values,
    })

    # Work for Collaborative Filtering
    users_items_pivot_matrix_df = interactions_df.pivot(index='user_id', 
                                                          columns='article_id', 
                                                          values='click').fillna(0)

    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    users_ids = list(users_items_pivot_matrix_df.index)
    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
    NUMBER_OF_FACTORS_MF = 15
    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns, index=users_ids).transpose()
    
    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')


    model = req.params.get('model')
    if model == 'popularity':
        recommender_model = RecommenderModels.PopularityRecommender(popularity_df, articles_df)
    elif model == 'content-based':
        recommender_model = RecommenderModels.ContentBasedRecommender(interactions_df, articles_df)
    elif model == 'collaborative-filtering':
        recommender_model = RecommenderModels.CFRecommender(cf_preds_df, articles_df)
    else:
        # Modèle non valide, retourner une erreur appropriée
        error_message = {"error": "Invalid model. Must be 'popularity', 'content-based', or 'collaborative-filtering'."}
        return func.HttpResponse(json.dumps(error_message), headers=headers, status_code=400, mimetype="application/json")

    print(model)

    if user_id:
        if user_id.isdigit():
            user_id = int(user_id)
            
            if model == 'popularity':
                if user_id in set(interactions_df.user_id):
                    user_history = list(set(interactions_df[interactions_df.user_id==user_id].article_id))
                else:
                    user_history = []
                recommended_items = list(recommender_model.recommend_items(user_id=user_id, items_to_ignore=user_history, topn=5).article_id)
            else:
                if user_id in set(interactions_df.user_id):
                    user_history = list(set(interactions_df[interactions_df.user_id==user_id].article_id))
                    if model == 'content-based':
                        recommended_items = list(recommender_model.recommend_items(user_id=user_id, topn=5).article_id)
                    else:
                        recommended_items = list(recommender_model.recommend_items(user_id=user_id, items_to_ignore=user_history, topn=5).article_id)
                else:
                    error_message = {"error": "The passed user_id not found."}
                    return func.HttpResponse(json.dumps(error_message), headers=headers, status_code=404, mimetype="application/json")
                
            response_data = {
                "user_id": user_id,
                "user_history": user_history,
                "recommended_items": recommended_items
            }
            response_json = json.dumps(response_data)

            return func.HttpResponse(response_json, headers=headers, mimetype="application/json")
                
        else:
            error_message = {
                "error": "Invalid user_id. Must be a number."
            }
            return func.HttpResponse(json.dumps(error_message), headers=headers, status_code=400, mimetype="application/json")
        
    error_message = {"error": "Pass a user_id in the query string or in the request body for a personalized response."}
    return func.HttpResponse(json.dumps(error_message), headers=headers, status_code=400, mimetype="application/json")