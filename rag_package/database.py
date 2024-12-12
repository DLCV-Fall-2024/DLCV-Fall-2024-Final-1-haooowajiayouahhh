import faiss
import sqlite3
import json
import numpy as np
import os

class FAISSDatabase:
    def __init__(self, index_path, metadata_path="metadata.db"):
        self.vit_emb = faiss.IndexFlatL2(768)
        self.obj_pre_vec = faiss.IndexFlatL2(34)
        self.vit_emb_index_path = os.path.join(index_path, "vit_emb.faiss")
        self.obj_pre_vec_index_path = os.path.join(index_path, "obj_pre_vec.faiss")
        self.conn = sqlite3.connect(metadata_path)
        self._create_metadata_table()


    def _create_metadata_table(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS metadata 
                                 (vector_id TEXT, image_id TEXT)''')

    def add_vit_emb_vector(self, vector_id, image_id, vector):
        self.vit_emb.add(vector)
        with self.conn:
            self.conn.execute("INSERT INTO metadata VALUES (?, ?)", (vector_id, image_id))

    def add_obj_pre_vec(self, vector_id, image_id, vector):
        self.obj_pre_vec.add(vector)
        with self.conn:
            self.conn.execute("INSERT INTO metadata VALUES (?,?)", (vector_id, image_id))
    
    def save(self):
        faiss.write_index(self.vit_emb, self.vit_emb_index_path)
        faiss.write_index(self.obj_pre_vec, self.obj_pre_vec_index_path)


class JSONDataProcessor:
    def __init__(self, json_path):
        self.json_path = json_path
        self.presence_vector_list = []
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

    
    def get_object_presence_vector(self):
        """
        Convert formatted output into binary vector indicating presence of each object type
    
        Args:
            formatted_output (dict): Dictionary containing detected objects by CODA categories
        
        Returns:
            list: Binary vector where 1 indicates object presence and 0 indicates absence
        """
    # Define all object categories from text
        categories = [
            'car', 'truck', 'bus', 'van', 'suv', 'trailer', 'construction vehicle', 'recreational vehicle',
            'pedestrian', 'cyclist', 'motorcycle', 'bicycle', 'tricycle', 'moped', 'wheelchair', 'stroller',
            'traffic sign', 'warning sign',
            'traffic light',
            'traffic cone',
            'barrier', 'bollard', 'concrete block',
            'traffic island', 'traffic box', 'debris', 'machinery', 'dustbin', 'cart', 'chair', 'basket', 'suitcase', 'dog', 'phone booth'
        ]
        for image_id in self.data.keys():
            if "regional" not in image_id:
                presence_vector = [0] * len(categories)
                for category_group in self.data[image_id].values():
                    for obj in category_group:
                        try:
                            idx = categories.index(obj['label'].lower())
                            presence_vector[idx] = 1
                        except ValueError:
                            print(f"MWarning: Unknown object label {obj['label']}")
                            continue
                presence_vector = np.array(presence_vector)
                self.presence_vector_list.append((presence_vector, image_id))

            




        