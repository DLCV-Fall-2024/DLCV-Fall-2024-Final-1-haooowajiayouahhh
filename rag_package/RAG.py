import os
import numpy as np
from rag_package import ImageEmbedder, FAISSDatabase, RAGQuery, JSONDataProcessor 
import rag_package.config as cfg


class RAG:
    def __init__(self, config, task):
        self.config = config
        self.task = task
        self.dataset_name = self.config["dataset_name"]
        self.model_name = self.config["model_name"]
        os.makedirs(self.config["FAISS_PATH"], exist_ok=True)
        os.makedirs(os.path.join(self.config["FAISS_PATH"], self.task), exist_ok=True)
        self.task_faiss_path = os.path.join(self.config["FAISS_PATH"], self.task) 
        self.metadata_path = os.path.join(self.config["FAISS_PATH"], self.task, "metadata.db") 
        self.vit_embedder = ImageEmbedder(self.dataset_name, self.model_name, self.config["test"], self.task) 
        self.json_data = JSONDataProcessor(self.config["JSON_PATH"], self.task)
        self.database = FAISSDatabase(self.task_faiss_path, self.metadata_path, self.task)
        if self.config["init"]:
            self.set_vit_embedding()
            if self.task != 'regional':
                self.set_obj_presence_vector()
            print(f"Metadata saved to {self.task_faiss_path}!")
            self.database.save()
            print(f"Encoding vectors saved to {self.task_faiss_path}!")

        self.vit_emb_query = RAGQuery(os.path.join(self.task_faiss_path, "vit_emb.faiss"), self.metadata_path)
        self.obj_pre_query = RAGQuery(os.path.join(self.task_faiss_path, "obj_pre_vec.faiss"), self.metadata_path)


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

    def vit_emb_query_image_by_image_id(self, image_id, k=5):
        #returns distances, indices of vectors, image_ids on huggingface
        return self.vit_emb_query.search_by_image_id(image_id, k)
    
    def obj_pre_query_image_by_image_id(self, image_id, k=5):
        return self.obj_pre_query.search_by_image_id(image_id, k)
        
    def vit_emb_query_image_by_vector(self, vector, k=5):
        if self.task == 'regional':
            return
        return self.vit_emb_query.search_by_vector(vector, k)

    def obj_pre_query_image_by_vector(self, vector, k=5):
        return self.obj_pre_query.search_by_vector(vector, k)
