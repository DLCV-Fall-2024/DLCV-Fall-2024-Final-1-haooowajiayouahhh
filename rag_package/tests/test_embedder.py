from rag_package.embedder import ImageEmbedder

def test_embedder():
    # Replace with your actual dataset name
    embedder = ImageEmbedder(test_mode = True)
    embeddings, ids = embedder.encode_images_with_vit()
    
    # Print some debug info
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Shape of first embedding: {embeddings[0].shape}")
    print(f"First few IDs: {ids[:5]}")
    # print(f"Sample metadata: {metadata[0]}")

