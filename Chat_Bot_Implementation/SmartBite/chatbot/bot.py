from langchain_openai import ChatOpenAI
from SmartBite.chatbot.chains.Personalized_Recipe_Suggestions_Chain import PersonalizedRecipeSuggestionsChain
from SmartBite.chatbot.chains.Ingredient_Based_Recipe_Chain import IngredientBasedRecipeChain
from SmartBite.chatbot.chains.Nutrition_Info_Response_Chain import NutritionInfoResponseChain
from SmartBite.chatbot.chains.Step_By_Step_Cooking_Chain import StepByStepCookingChain
from SmartBite.chatbot.chains.Nutrition_Goal_Sorting_Chain import NutritionGoalSortingChain
from SmartBite.chatbot.chains.Recipe_Difficulty_Filter_Chain import RecipeDifficultyFilterChain
from SmartBite.chatbot.chains.Ingredient_Update_Chain import IngredientUpdateChain
from SmartBite.chatbot.chains.Save_Favorite_Recipe_Chain import SaveFavoriteRecipeChain
from SmartBite.chatbot.chains.About_Company_Chain import AboutCompanyChain
from SmartBite.chatbot.chains.Small_Talk_Chain import SmallTalkChain
from SmartBite.chatbot.memory import MemoryManager
from SmartBite.chatbot.router.loader import load_intention_classifier


class SmartBiteBot:
    """
    A chatbot class for SmartBite to handle user interactions and provide tailored responses.

    Attributes:
        memory (MemoryManager): Manages conversation memory.
        user_id (str): The unique identifier for the user.
        conversation_id (str): The unique identifier for the conversation.
        memory_config (dict): Configuration for user and conversation-specific memory.
        similarity_threshold (float): Minimum similarity score for intent classification.
        llm (ChatOpenAI): The language model used for generating responses.
        chain_map (dict): Mapping of intents to corresponding chains.
        intention_classifier: Classifier for determining user intent.
    """

    def __init__(self, user_id: str, conversation_id: str, similarity_threshold=0.6):

        """
        Initialize the bot with session and language model configurations.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.
            similarity_threshold (float): Minimum similarity score for a valid intent classification. Defaults to 0.6.
        """

        self.memory = MemoryManager()
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_config = {"user_id": self.user_id, "conversation_id": self.conversation_id}
        self.similarity_threshold = similarity_threshold

        # Configure language model
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

        # Path to the recipe JSON file
        recipe_file_path = "SmartBite/chatbot/data/Data_Collection/processed_data.json"
        pdf_paths = [
            "SmartBite/chatbot/data/Pdf's/PDF1_Operational_Overview_and_Features_Explanation.pdf",
            "SmartBite/chatbot/data/Pdf's/PDF2_User_Manual_and_Support_Resources.pdf",
            "SmartBite/chatbot/data/Pdf's/PDF3_Feature_Highlights_and_Advanced_Capabilities.pdf",
        ]

        # Chains map
        self.chain_map = {
            "personalized_recipe": PersonalizedRecipeSuggestionsChain(llm=self.llm, recipe_file_path=recipe_file_path),
            "ingredient_based_recipe": IngredientBasedRecipeChain(llm=self.llm, recipe_file_path=recipe_file_path),
            "nutrition_info": NutritionInfoResponseChain(llm=self.llm, recipe_file_path=recipe_file_path),
            "step_by_step_instruction": StepByStepCookingChain(llm=self.llm, recipe_file_path=recipe_file_path),
            "nutrition_goal_sorting": NutritionGoalSortingChain(llm=self.llm, recipe_file_path=recipe_file_path),
            "recipe_difficulty_filter": RecipeDifficultyFilterChain(llm=self.llm, recipe_file_path=recipe_file_path),
            "ingredient_update": IngredientUpdateChain(),
            "save_favorite_recipe": SaveFavoriteRecipeChain(),
            "about_company": AboutCompanyChain(pdf_paths=pdf_paths),
            "None_related": SmallTalkChain(llm=self.llm),
        }

        # Load intention classifier
        self.intention_classifier = load_intention_classifier()

    def classify_intent(self, user_input: str):
        """
        Classify user intent using the intention classifier.

        Args:
            user_input (str): Input text from the user.

        Returns:
            tuple: A tuple containing:
                - intent (str): The classified intent of the user's input.
                - similarity_score (float): The confidence score for the classified intent.
        """

        try:
            routes = self.intention_classifier.retrieve_multiple_routes(user_input)
            if routes:
                top_route = routes[0]  # Get the top matching route
                return top_route.name, top_route.similarity_score
            else:
                return None, 0.0
        except Exception as e:
            self.memory.add_message_to_history(
                self.user_id, self.conversation_id, "bot", f"Error classifying intent: {e}"
            )
            return None, 0.0

    def handle_intent(self, intent: str, user_input: str) -> str:
        """
        Handle the user's intent by routing to the appropriate chain.

        Args:
            intent (str): The classified intent of the user's input.
            user_input (str): The input text from the user.

        Returns:
            str: A response string from the appropriate chain, or a fallback message if the intent is not recognized.
        """

        chain = self.chain_map.get(intent)
        if not chain:
            fallback_message = "Sorry, I couldn’t understand your question. Could you rephrase it?"
            self.memory.add_message_to_history(self.user_id, self.conversation_id, "bot", fallback_message)
            return fallback_message

        try:
            # Pass the user_id for chains that require it
            if intent in ["ingredient_update", "save_favorite_recipe"]:
                result = chain.run(user_input, user_id=self.user_id)
            else:
                result = chain.run(user_input)

            self.memory.add_message_to_history(self.user_id, self.conversation_id, "bot", result)
            return result
        except Exception as e:
            error_message = f"Error in chain '{intent}': {e}"  # Add chain-specific error
            print(error_message)  # Log the error for debugging
            self.memory.add_message_to_history(self.user_id, self.conversation_id, "bot", error_message)
            return "An error occurred while processing your request. Please try again."



    def process_input(self, user_input: str) -> str:
        """
        Process the user's input by classifying intent and invoking the appropriate handler.

        Args:
            user_input (str): The input text from the user.

        Returns:
            str: The bot's response based on the classified intent or a fallback message.
        """

        self.memory.add_message_to_history(self.user_id, self.conversation_id, "user", user_input)
        intent, similarity_score = self.classify_intent(user_input)
        print(f"Classified Intent: {intent}, Similarity Score: {similarity_score}")  # Debugging line

        # Check if the intent meets the similarity threshold
        if intent and similarity_score >= self.similarity_threshold:
            return self.handle_intent(intent, user_input)
        else:
            fallback_message = "Sorry, I couldn’t understand your question. Could you rephrase it?"
            self.memory.add_message_to_history(self.user_id, self.conversation_id, "bot", fallback_message)
            return fallback_message