# Import necessary libraries and modules
from SmartBite.chatbot.chains.base import ChainBase

class SmallTalkChain(ChainBase):
    """
    Chain for engaging in casual small talk with users.
    """

    def __init__(self, llm):
        """
        Initialize the SmallTalkChain with a language learning model (LLM).

        Args:
            llm (ChatOpenAI): The language model to generate responses for small talk.
        """

        super().__init__(
            system_template="""
            You are a friendly, humorous conversational partner for a virtual assistant chatbot. Your task is to engage users in casual small talk, keeping the conversation light and enjoyable.

            Avoid technical jargon, formal language, or overly complex responses. If the input is unclear or inappropriate, respond with a polite, generic message.

            Examples:
            - User: "Hi, how are you?"
              Response: "I'm doing great, thanks for asking!"
            - User: "Tell me a joke."
              Response: "Why did the computer get cold? It left its Windows open!"
            - User: "What’s your favorite color?"
              Response: "I’d say blue—it reminds me of the sky!"
            - User: [inappropriate input]
              Response: "Let’s keep things friendly, shall we?"

            At the end of your response, gently remind the user of your main role as a SmartBite assistant specializing in recipes, ingredients, and meal planning.
            """,
            human_template="""
            User Input: {user_input}
            """,
            memory=False,
        )
        self.llm = llm

    def run(self, user_input: str):
        """
        Run the chain to generate a response for small talk.

        Args:
            user_input (str): User's input for casual conversation.

        Returns:
            str: A response generated for the small talk, or a fallback message if no response is generated.
        """

        inputs = {
            "user_input": user_input
        }

        # Call the LLM and process the response
        response = self.llm.invoke([
            {"role": "system", "content": self.prompt_template.system_template},
            {"role": "user", "content": inputs["user_input"]}
        ])

        # Return the response or a fallback message
        if response and response.content:
            return response.content
        else:
            return "Sorry, I didn’t understand. Can you try rephrasing?"