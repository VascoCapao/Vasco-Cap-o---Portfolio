from SmartBite.chatbot.chains.base import ChainBase
from SmartBite.chatbot.data.AboutSmartbite_extract import CompanyKnowledgeBase

class AboutCompanyChain(ChainBase):
    """
    Chain to handle queries about the company.
    Uses the CompanyKnowledgeBase to retrieve information from indexed PDFs.
    """

    def __init__(self, pdf_paths, embedding_model="text-embedding-ada-002", k=1):
        """
        Initialize the chain with the knowledge base and query settings.

        Args:
            pdf_paths (list[str]): List of paths to the company's PDF documents.
            embedding_model (str): The embedding model to use.
            k (int): Number of top results to return.
        """
        super().__init__(
            system_template="""
            You are a virtual assistant specialized in answering questions about SmartBite as a company.
            Use the provided company knowledge base to answer questions accurately and concisely.

            Question: {question}

            If the knowledge base does not contain relevant information, respond with:
            "I'm sorry, I couldn't find information about that in the company's knowledge base."
            """,
            human_template="""
            User Question: {user_input}
            """,
            memory=False,  # This chain does not rely on conversational history.
        )

        self.knowledge_base = CompanyKnowledgeBase(pdf_paths, embedding_model=embedding_model)
        self.k = k

        # Index the PDFs or load the existing index
        try:
            self.knowledge_base.index_pdfs()
        except RuntimeError as e:
            print(f"Error initializing knowledge base: {e}")

    def process_input(self, user_input: str) -> dict:
        """
        Prepare the input for querying the knowledge base.

        Args:
            user_input (str): The user's question.

        Returns:
            dict: A dictionary with the formatted input for the knowledge base query, 
                  where the key 'question' maps to the user's input string.
        """
        return {"question": user_input}

    def run(self, user_input: str) -> str:
        """
        Run the chain to process the user's input and return the result.

        Args:
            user_input (str): The user's question.

        Returns:
            str: The response to the user's question.
        """
        # Query the knowledge base
        try:
            results = self.knowledge_base.query(user_input, k=self.k)
            if not results:
                return "I'm sorry, I couldn't find information about that in the company's knowledge base."

            # Combine the top results into a single response
            response = "\n\n".join(
                [result.page_content if hasattr(result, "page_content") else result for result in results]
            )
            return response
        except ValueError as e:
            return "Error querying knowledge base: The knowledge base is not properly initialized."
        except Exception as e:
            return f"An unexpected error occurred while processing your request: {e}"
