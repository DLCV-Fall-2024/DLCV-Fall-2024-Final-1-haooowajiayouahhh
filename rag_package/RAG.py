import os
import numpy as np
from rag_package import ImageEmbedder, FAISSDatabase, RAGQuery
import rag_package.config as cfg


class RAG:
    def __init__(self, config):
        self.config = config
        self.dataset_name = self.config["dataset_name"]
        self.model_name = self.config["model_name"]
        self.embedder = ImageEmbedder(self.dataset_name, self.model_name, test_mode = self.config["testing"])
        self.database = FAISSDatabase(self.config["FAISS_INDEX_PATH"], self.config["EMBEDDING_DIM"])
        self.query = RAGQuery(config["FAISS_INDEX_PATH"], config["METADATA_PATH"])
        all_embeddings, all_ids = self.embedder.encode_images_with_vit()
        for idx, (embedding, image_id) in enumerate(zip(all_embeddings, all_ids)):
            embedding = np.expand_dims(embedding, axis=0)
            self.database.add_vector(idx, image_id, embedding)
        print(f"Metadata saved to {os.path.join(os.getcwd(), self.config['METADATA_PATH'])}!")
        self.database.save()
        print(f"Encoding vectors saved to {config['FAISS_INDEX_PATH']}!")

    def query_image(self, image_id, k=5):
        #returns distances, indices of vectors, image_ids on huggingface
        return self.query.search(image_id, k)
        

