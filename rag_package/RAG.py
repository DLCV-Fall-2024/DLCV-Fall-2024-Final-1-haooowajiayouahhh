import os
import numpy as np
from rag_package import ImageEmbedder, FAISSDatabase, RAGQuery, JSONDataProcessor 
import rag_package.config as cfg


class RAG:
    def __init__(self, config):
        self.config = config
        self.dataset_name = self.config["dataset_name"]
        self.model_name = self.config["model_name"]
        self.vit_embedder = ImageEmbedder(self.dataset_name, self.model_name, test_mode = self.config["testing"])
        self.json_data = JSONDataProcessor(self.config["JSON_PATH"])
        self.database = FAISSDatabase(self.config["FAISS_PATH"])
        self.set_vit_embedding()
        self.set_obj_presence_vector()

        print(f"Metadata saved to {os.path.join(os.getcwd(), self.config['METADATA_PATH'])}!")
        self.database.save()
        print(f"Encoding vectors saved to {config['FAISS_PATH']}!")

        self.vit_emb_query = RAGQuery(config["FAISS_PATH"]+"vit_emb.faiss", config["METADATA_PATH"])
        self.obj_pre_query = RAGQuery(config["FAISS_PATH"]+"obj_pre_vec.faiss", config["METADATA_PATH"])


    def set_vit_embedding(self):
        all_embeddings, all_ids = self.vit_embedder.encode_images_with_vit()
        for idx, (embedding, image_id) in enumerate(zip(all_embeddings, all_ids)):
            embedding = np.expand_dims(embedding, axis=0)
            self.database.add_vit_emb_vector(idx, image_id, embedding)

    def set_obj_presence_vector(self):
        self.json_data.get_object_presence_vector()
        for idx, (presence_vector, image_id) in enumerate(self.json_data.presence_vector_list):
            presence_vector = np.expand_dims(presence_vector, axis=0)
            self.database.add_obj_pre_vec(idx, image_id, presence_vector)

    def vit_emb_query_image(self, image_id, k=5):
        #returns distances, indices of vectors, image_ids on huggingface
        return self.vit_emb_query.search(image_id, k)
    
    def obj_pre_query_image(self, image_id, k=5):
        return self.obj_pre_query.search(image_id, k)
        
