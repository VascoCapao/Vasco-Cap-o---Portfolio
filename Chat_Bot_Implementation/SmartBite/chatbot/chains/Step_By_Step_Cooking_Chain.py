import json
from SmartBite.chatbot.chains.base import ChainBase

class StepByStepCookingChain(ChainBase):
    """
    Chain to provide step-by-step instructions for preparing a recipe.
    """

    def __init__(self, llm, recipe_file_path):
        """
        Initialize the StepByStepCookingChain with a language learning model (LLM) and a recipe JSON file.

        Args:
            llm (ChatOpenAI): The language model for processing user queries.
            recipe_file_path (str): Path to the JSON file containing recipe data.
        """

        super().__init__(
            system_template="""
            You are a culinary assistant that provides step-by-step instructions for recipes.
            Your task is to retrieve and present the cooking steps for a specific recipe from the provided database.

            Recipe Database:
            {recipe_database}

            Ensure your response includes only:
            - The recipe title
            - All cooking steps presented sequentially.
            """,
            human_template="""
            User Input: {user_input}
            """,
            memory=False,
        )
        self.llm = llm
        self.recipe_file_path = recipe_file_path
        self.recipe_database = self._load_recipes()

    def _load_recipes(self):
        """
        Load recipes from a JSON file.

        Returns:
            list[dict]: A list of recipes, where each recipe is represented as a dictionary.
        """

        try:
            with open(self.recipe_file_path, "r") as file:
                data = json.load(file)
            print(f"Loaded {len(data)} recipes.")
            return data
        except FileNotFoundError:
            print(f"Error: File {self.recipe_file_path} not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

    def process_input(self, user_input: str):
        """
        Prepare input data by filtering recipes matching the user query.

        Args:
            user_input (str): User's input specifying the recipe title.

        Returns:
            dict or None: A dictionary containing the filtered recipes and user input, 
                        or None if no matching recipe is found.
        """

        user_query = user_input.strip().lower()
        filtered_recipes = []

        for recipe in self.recipe_database:
            if recipe["recipe_title"].lower() in user_query:
                filtered_recipes.append(recipe)
                break

        if filtered_recipes:
            return {
                "recipe_database": json.dumps(filtered_recipes, indent=2),
                "user_input": user_input,
                "filtered_recipes": filtered_recipes
            }
        else:
            return None

    def validate_response(self, response_content, filtered_recipes):
        """
        Validate that the response includes the steps for the requested recipe.

        Args:
            response_content (str): The content of the response to validate.
            filtered_recipes (list[dict]): The list of recipes filtered based on user input.

        Returns:
            bool: True if the response includes valid cooking steps, False otherwise.
        """

        valid_titles = {recipe["recipe_title"].lower() for recipe in filtered_recipes}
        for title in valid_titles:
            if title in response_content.lower():
                return True
        return False

    def format_response(self, filtered_recipes):
        """
        Format the cooking steps in a user-friendly way.

        Args:
            filtered_recipes (list[dict]): The list of filtered recipes to format.

        Returns:
            str: A formatted string containing the recipe title and step-by-step instructions.
        """

        formatted_response = []
        for recipe in filtered_recipes:
            recipe_info = f"### {recipe['recipe_title']}\n"
            for step in recipe.get("cooking_steps", []):
                recipe_info += f"{step[1]}\n"
            formatted_response.append(recipe_info)
        return "\n".join(formatted_response)

    def run(self, user_input: str):
        """
        Run the chain to get step-by-step instructions for a recipe.

        Args:
            user_input (str): User's input specifying the recipe title.

        Returns:
            str: A response containing step-by-step instructions or an error message if no matches are found.
        """

        inputs = self.process_input(user_input)

        if not inputs:
            return "Sorry, I couldn't find the recipe you're looking for."

        # Structure messages for the LLM
        messages = [
            {"role": "system", "content": inputs["recipe_database"]},
            {"role": "user", "content": inputs["user_input"]}
        ]

        # Call the ChatOpenAI model and process the response
        response = self.llm.invoke(messages)

        # Validate the response
        filtered_recipes = inputs["filtered_recipes"]
        if self.validate_response(response.content, filtered_recipes):
            return self.format_response(filtered_recipes)
        else:
            return "Sorry, I couldn't find the recipe you're looking for."