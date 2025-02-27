from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class CompanyKnowledgeBase:
    def __init__(self, pdf_paths: list[str], embedding_model="text-embedding-ada-002", index_file="faiss_index"):
        """
        Initialize the knowledge base with the paths to the PDFs and the embedding model.

        Args:
            pdf_paths (list[str]): List of file paths to the company's PDF documents.
            embedding_model (str): OpenAI embedding model to use.
            index_file (str): Path to save/load the FAISS index file.
        """
        self.pdf_paths = pdf_paths
        self.embedding_model = embedding_model
        self.index_file = index_file
        self.vectorstore = None

    def index_pdfs(self):
        """
        Load PDFs, extract content, and create a searchable index.

        This method processes the provided PDF documents, splits their content into manageable chunks, 
        and creates a FAISS index for similarity search.

        Raises:
            RuntimeError: If there is an error while loading or processing the PDFs.
        """

        try:
            documents = []
            for path in self.pdf_paths:
                loader = PyMuPDFLoader(path)  # Use PyMuPDFLoader to load PDFs
                docs = loader.load()  # Extract content
                documents.extend(docs)  # Add to the list of documents

            # Split documents into chunks for better embedding handling
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(model=self.embedding_model)  # Load embedding model
            self.vectorstore = FAISS.from_documents(chunks, embeddings)  # Create FAISS index

            # Save the index locally
            self.vectorstore.save_local(self.index_file)
            print("PDFs indexed successfully and saved locally!")
        except Exception as e:
            raise RuntimeError(f"Error while indexing PDFs: {e}")

    def load_index(self):
        """
        Load the FAISS index from the local file.

        This method loads a previously saved FAISS index, allowing similarity searches 
        to be performed without re-indexing the documents.

        Raises:
            RuntimeError: If there is an error while loading the FAISS index.
        """

        try:
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
            self.vectorstore = FAISS.load_local(
                self.index_file, embeddings, allow_dangerous_deserialization=True
            )
            print("FAISS index loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Error while loading FAISS index: {e}")


    def query(self, question: str, k=1):
        """
        Query the indexed knowledge base and return the best result.

        Args:
            question (str): The user's question to query the knowledge base.
            k (int): The number of top results to retrieve. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - str: The content of the most relevant document.
                - str: The metadata of the source document (e.g., file name or path).

        Raises:
            ValueError: If the knowledge base is not indexed.
        """

        if not self.vectorstore:
            raise ValueError("Knowledge base is not indexed. Call `index_pdfs` or `load_index` first.")

        # Perform the similarity search
        results = self.vectorstore.similarity_search(question, k=k)
        
        if results:
            # Return the top result's content and metadata
            top_result = results[0]
            return top_result.page_content, top_result.metadata.get("source", "Unknown source")
        else:
            return "No relevant information found.", "N/A"