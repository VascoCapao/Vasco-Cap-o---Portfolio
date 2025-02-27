# Import necessary libraries and modules
import json
from SmartBite.chatbot.chains.base import ChainBase

class IngredientBasedRecipeChain(ChainBase):
    """
    Chain to suggest recipes based on ingredients available to the user.
    """

    def __init__(self, llm, recipe_file_path):
        """Initialize the chain with an LLM and a recipe JSON file."""
        super().__init__(
            system_template="""
            You are a culinary assistant specializing in suggesting recipes based on available ingredients.
            You must only suggest recipes from the provided database. Do not create new recipes or use external knowledge.

            Recipe Database:
            {recipe_database}

            Ensure your response is clear and includes the name of the recipe, required ingredients, and brief instructions.
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
        """Load recipes from a JSON file.

        Returns:
            list[dict]: A list of recipes where each recipe is represented as a dictionary.
        """

        with open(self.recipe_file_path, "r") as file:
            data = json.load(file)
        print(f"Loaded {len(data)} recipes.")
        return data

    def process_input(self, user_input: str, max_recipes=10):
        """Prepare input data by filtering recipes with at least one matching ingredient.

        Args:
            user_input (str): A string of ingredients provided by the user, separated by spaces.
            max_recipes (int): The maximum number of recipes to return. Defaults to 10.

        Returns:
            dict or None: A dictionary with filtered recipes and user input if matches are found, 
                        or None if no recipes match the provided ingredients.
        """

        # Normalize user-provided ingredients
        user_ingredients = set(ingredient.strip().lower() for ingredient in user_input.split())
        print("User Ingredients:", user_ingredients)

        filtered_recipes = []

        for recipe in self.recipe_database:
            # Normalize the ingredients in the recipe
            recipe_ingredients = {
                ingredient["ingredient_name"].strip().lower()
                for ingredient in recipe.get("ingredients", [])
                if ingredient.get("ingredient_name")
            }

            # Check if at least one user-provided ingredient is in the recipe's ingredients
            match_found = any(
                user_ing in recipe_ing for user_ing in user_ingredients for recipe_ing in recipe_ingredients
            )

            if match_found:
                recipe["matched_ingredients"] = sum(
                    1 for user_ing in user_ingredients for recipe_ing in recipe_ingredients if user_ing in recipe_ing
                )
                filtered_recipes.append(recipe)

        # Sort recipes by the number of matching ingredients
        filtered_recipes.sort(key=lambda x: x["matched_ingredients"], reverse=True)

        # Limit the number of recipes to the maximum allowed
        limited_recipes = filtered_recipes[:max_recipes]

        # Return formatted input
        if limited_recipes:
            return {
                "recipe_database": json.dumps(limited_recipes, indent=2),
                "user_input": user_input
            }
        else:
            return None

    def validate_response(self, response_content, filtered_recipes):
        """Validate that the response includes a recipe from the filtered database.

        Args:
            response_content (str): The content of the response to validate.
            filtered_recipes (list[dict]): The list of recipes filtered based on user input.

        Returns:
            bool: True if the response includes a valid recipe title from the filtered recipes, 
                False otherwise.
        """
        valid_titles = {recipe["recipe_title"].lower() for recipe in filtered_recipes}
        for title in valid_titles:
            if title in response_content.lower():
                return True
        return False

    def run(self, user_input: str):
        """Run the chain to get recipe suggestions based on ingredients.

        Args:
            user_input (str): A string of ingredients provided by the user, separated by spaces.

        Returns:
            str: A response message containing suggested recipes or an error message if no matches are found.
        """
        inputs = self.process_input(user_input)

        if not inputs:
            return "Sorry, no recipes match the provided ingredients."

        # Correct message structure
        messages = [
            {"role": "system", "content": inputs["recipe_database"]},
            {"role": "user", "content": inputs["user_input"]}
        ]

        # Call the ChatOpenAI model and process the response
        response = self.llm.invoke(messages)

        # Validate if the response includes a valid recipe
        filtered_recipes = json.loads(inputs["recipe_database"])
        if self.validate_response(response.content, filtered_recipes):
            return response.content
        else:
            return "Sorry, no valid recipes were found in the database."