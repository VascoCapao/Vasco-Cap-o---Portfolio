import json
from SmartBite.chatbot.chains.base import ChainBase


class RecipeDifficultyFilterChain(ChainBase):
    """
    Chain to filter and present recipes based on difficulty level.
    """

    def __init__(self, llm, recipe_file_path):
        """
        Initialize the RecipeDifficultyFilterChain with an LLM and a recipe JSON file.

        Args:
            llm (ChatOpenAI): The language model for processing queries.
            recipe_file_path (str): Path to the JSON file containing recipe data.
        """

        super().__init__(
            system_template="""
            You are a culinary assistant specializing in recipe suggestions based on difficulty levels.
            Filter recipes from the provided database based on the specified difficulty level.

            Recipe Database:
            {recipe_database}

            Ensure your response includes only:
            - The difficulty level ('a challenge', 'more effort', 'easy', or None)
            - Up to 15 recipes matching the difficulty
            - Present the recipes as a simple bulleted list.
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

    def process_input(self, user_input: str, max_recipes=15):
        """
        Prepare input data by filtering recipes based on difficulty level.

        Args:
            user_input (str): User's input specifying the desired difficulty level.
            max_recipes (int): Maximum number of recipes to return. Defaults to 15.

        Returns:
            dict or None: A dictionary containing filtered recipes, recipes without difficulty levels, 
                        and the selected difficulty, or None if no valid difficulty is found.
        """

        synonym_mapping = {
            "easy": "easy",
            "simple": "easy",
            "hard": "a challenge",
            "difficult": "a challenge",
            "medium": "more effort",
        }

        user_query = user_input.strip().lower()
        difficulty = next(
            (synonym_mapping[key] for key in synonym_mapping if key in user_query),
            None,
        )

        if not difficulty:
            return None  # No valid difficulty level found in the user query

        # Separate recipes with difficulty None
        recipes_without_difficulty = [
            recipe["recipe_title"] for recipe in self.recipe_database
            if recipe.get("extra_info", {}).get("difficulty") is None
        ]

        # Filter recipes based on the difficulty level
        filtered_recipes = [
            recipe["recipe_title"]
            for recipe in self.recipe_database
            if recipe.get("extra_info", {}).get("difficulty") and
            recipe.get("extra_info", {}).get("difficulty", "").lower() == difficulty
        ]

        # Limit the number of recipes
        filtered_recipes = filtered_recipes[:max_recipes]


        # Return input data for the chain
        return {
            "recipe_database": json.dumps(filtered_recipes, indent=2),
            "user_input": user_input,
            "filtered_recipes": filtered_recipes,
            "recipes_without_difficulty": recipes_without_difficulty,
            "difficulty": difficulty.capitalize(),
        }

    def format_response(self, filtered_recipes, recipes_without_difficulty, difficulty):
        """
        Format the response with the recipe titles.

        Args:
            filtered_recipes (list[str]): List of recipe titles filtered by difficulty.
            recipes_without_difficulty (list[str]): List of recipes without a specified difficulty level.
            difficulty (str): The selected difficulty level.

        Returns:
            str: A formatted string containing recipes by difficulty and recipes without a specified difficulty level.
        """

        formatted_response = f"### {difficulty} Recipes\n"
        if filtered_recipes:
            for recipe in filtered_recipes:
                formatted_response += f"- {recipe}\n"
        else:
            formatted_response += "No recipes found for the selected difficulty level.\n"

        if recipes_without_difficulty:
            formatted_response += (
                "\n### Recipes Without Difficulty Level\n"
                "The following recipes do not have a specified difficulty level:\n"
            )
            for recipe in recipes_without_difficulty:
                formatted_response += f"- {recipe}\n"

        return formatted_response

    def run(self, user_input: str):
        """
        Run the chain to get recipes filtered by difficulty.

        Args:
            user_input (str): User's input specifying the desired difficulty level.

        Returns:
            str: A response containing recipes filtered by difficulty or an error message if no matches are found.
        """

        inputs = self.process_input(user_input)

        if not inputs:
            return "Sorry, I couldn't find recipes matching the specified difficulty level."

        # Structure messages for the LLM
        messages = [
            {"role": "system", "content": inputs["recipe_database"]},
            {"role": "user", "content": inputs["user_input"]}
        ]

        # Call the ChatOpenAI model and process the response
        response = self.llm.invoke(messages)

        # Validate and format the response
        filtered_recipes = inputs["filtered_recipes"]
        recipes_without_difficulty = inputs["recipes_without_difficulty"]
        difficulty = inputs["difficulty"]
        return self.format_response(filtered_recipes, recipes_without_difficulty, difficulty)