class recommender_statistics:
    def __init__(self, rating_matrix, similarity_matrix, user_to_user_id, item_to_item_id):
        self.rating_matrix = rating_matrix
        self.similarity_matrix = similarity_matrix
        self.user_to_user_id = user_to_user_id
        self.item_to_item_id = item_to_item_id