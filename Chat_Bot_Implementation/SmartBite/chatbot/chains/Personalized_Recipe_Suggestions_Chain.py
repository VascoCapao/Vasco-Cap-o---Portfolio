import json
from SmartBite.chatbot.chains.base import ChainBase

class PersonalizedRecipeSuggestionsChain(ChainBase):
    """
    Chain to provide personalized recipe suggestions based on dietary requirements and preferences.
    """

    def __init__(self, llm, recipe_file_path):
        """
        Initialize the PersonalizedRecipeSuggestionsChain with an LLM and a recipe JSON file.

        Args:
            llm (ChatOpenAI): The language model for processing queries.
            recipe_file_path (str): Path to the JSON file containing recipe data.
        """

        super().__init__(
            system_template="""
            You are a culinary assistant specializing in personalized recipe recommendations.
            Filter recipes from the provided database based on the user's dietary preferences or requirements.

            Recipe Database:
            {recipe_database}

            Ensure your response includes only:
            - A list of recipes matching the user's preferences.
            - If no recipes match, respond with a clear message indicating no results were found.
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

    def process_input(self, user_input: str, max_recipes=10):
        """
        Filter recipes based on user preferences or requirements.

        Args:
            user_input (str): User's input specifying dietary preferences or requirements.
            max_recipes (int): Maximum number of recipes to return. Defaults to 10.

        Returns:
            dict or None: A dictionary containing filtered recipes and user input, 
                        or None if no matching recipes are found.
        """

        # Normalize user query and extract preferences
        user_query = user_input.strip().lower()
        user_preferences = {pref.strip() for pref in user_query.split() if pref in [
            "dairy-free", "easily doubled", "easily halved", "egg-free", "freezable",
            "gluten-free", "healthy", "high-fibre", "high-protein", "keto",
            "low calorie", "low carb", "low fat", "low sugar", "nut-free",
            "vegan", "vegetarian"]}

        filtered_recipes = []

        for recipe in self.recipe_database:
            recipe_tags = {tag.lower() for tag in recipe.get("recipe_tags", [])}

            # Check if all user preferences are satisfied by the recipe tags
            if user_preferences.issubset(recipe_tags):
                filtered_recipes.append(recipe)

        # Limit the number of recipes
        limited_recipes = filtered_recipes[:max_recipes]

        if limited_recipes:
            return {
                "recipe_database": json.dumps(limited_recipes, indent=2),
                "user_input": user_input,
                "filtered_recipes": limited_recipes
            }
        else:
            return None

    def format_response(self, filtered_recipes):
        """
        Format the response with the recipe titles.

        Args:
            filtered_recipes (list[dict]): The list of filtered recipes to format.

        Returns:
            str: A formatted string containing the recipe titles.
        """

        formatted_response = "### Personalized Recipes\n"
        for recipe in filtered_recipes:
            formatted_response += f"- {recipe['recipe_title']}\n"
        return formatted_response

    def run(self, user_input: str):
        """
        Run the chain to get personalized recipe suggestions.

        Args:
            user_input (str): User's input specifying dietary preferences or requirements.

        Returns:
            str: A response containing personalized recipe suggestions or an error message if no matches are found.
        """

        inputs = self.process_input(user_input)

        if not inputs:
            return "Sorry, no recipes match your preferences."

        # Structure messages for the LLM
        messages = [
            {"role": "system", "content": inputs["recipe_database"]},
            {"role": "user", "content": inputs["user_input"]}
        ]

        # Call the ChatOpenAI model and process the response
        response = self.llm.invoke(messages)

        # Validate and format the response
        filtered_recipes = inputs["filtered_recipes"]
        return self.format_response(filtered_recipes)