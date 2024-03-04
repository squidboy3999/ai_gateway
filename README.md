# ai_gateway
Is several services working togther so that api calls can be made to backend AI services. At present these services are:
- LLM service for text to text calls using a prespecified model.
- Document handler service which handles splitting documents into chunks, these may be used directly for summarization or embedded for vector operations.
- Embed model service converts text into vectors or converts vectors back into - uses a vector database for persistent embeddings
    - Vector database service might be only used with the embedding service unless other vector use cases emerge.
- Composer service can combine calls to back end services to perform common actions such as summarize a document, cluster common ideas within documents, retrieving useful embedding, and performing full rag functions.

