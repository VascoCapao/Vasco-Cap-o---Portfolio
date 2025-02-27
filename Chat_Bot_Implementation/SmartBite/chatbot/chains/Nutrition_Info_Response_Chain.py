# Import necessary libraries and modules
import json
from SmartBite.chatbot.chains.base import ChainBase

class NutritionInfoResponseChain(ChainBase):
    """
    Chain to provide detailed nutritional information for selected recipes.
    """

    def __init__(self, llm, recipe_file_path):
        """
        Initialize the NutritionInfoResponseChain with an LLM and a recipe JSON file.

        Args:
            llm (ChatOpenAI): The language model for processing queries.
            recipe_file_path (str): Path to the JSON file containing recipe data.
        """

        super().__init__(
            system_template="""
            You are a culinary assistant specializing in providing detailed nutritional information for recipes.
            Your task is to retrieve and present only the complete nutritional data for specific recipes from the provided database.
            Only use the recipes and information in the database. Do not include cooking steps, ingredients, or tags.

            Recipe Database:
            {recipe_database}

            Ensure your response includes only:
            - The recipe title
            - A list of nutritional categories, values, and units.
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

    def process_input(self, user_input: str, max_recipes=1):
        """
        Prepare input data by filtering recipes matching the user query.

        Args:
            user_input (str): User's query string containing the recipe title.
            max_recipes (int): Maximum number of recipes to return. Defaults to 1.

        Returns:
            dict or None: A dictionary containing filtered recipes and user input, or None if no matching recipe is found.
        """

        # Normalize user query
        user_query = user_input.strip().lower()

        # Attempt to extract recipe title from natural language input
        for recipe in self.recipe_database:
            if recipe["recipe_title"].lower() in user_query:
                filtered_recipes = [recipe]
                return {
                    "recipe_database": json.dumps(filtered_recipes, indent=2),
                    "user_input": recipe["recipe_title"],
                    "filtered_recipes": filtered_recipes
                }

        return None  # No matching recipe found


    def validate_response(self, response_content, filtered_recipes):
        """
        Validate that the response includes the nutritional data for a recipe.

        Args:
            response_content (str): The content of the response to validate.
            filtered_recipes (list[dict]): The list of recipes filtered based on user input.

        Returns:
            bool: True if the response includes valid nutritional data, False otherwise.
        """

        valid_titles = {recipe["recipe_title"].lower() for recipe in filtered_recipes}
        for title in valid_titles:
            if title in response_content.lower():
                return True
        return False

    def format_response(self, filtered_recipes):
        """
        Format the nutritional information in a user-friendly way.

        Args:
            filtered_recipes (list[dict]): The list of filtered recipes to format.

        Returns:
            str: A formatted string containing the recipe title and its nutritional information.
        """

        formatted_response = []
        for recipe in filtered_recipes:
            recipe_info = f"### {recipe['recipe_title']}\n"
            recipe_info += "**Nutritional Information:**\n"
            for nutrient in recipe.get("nutrition_info", []):
                recipe_info += f"- **{nutrient['category'].capitalize()}**: {nutrient['value']} {nutrient['unit']}\n"
            formatted_response.append(recipe_info)
        return "\n".join(formatted_response)

    def run(self, user_input: str):
        """
        Run the chain to get nutritional information for a specific recipe.

        Args:
            user_input (str): User's query string containing the recipe title.

        Returns:
            str: A response containing the nutritional information or an error message if no matches are found.
        """

        inputs = self.process_input(user_input)

        if not inputs:
            return "Sorry, no recipes match the provided query."

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
            # Format the response to be user-friendly
            return self.format_response(filtered_recipes)
        else:
            return "Sorry, no valid nutritional information was found in the database."