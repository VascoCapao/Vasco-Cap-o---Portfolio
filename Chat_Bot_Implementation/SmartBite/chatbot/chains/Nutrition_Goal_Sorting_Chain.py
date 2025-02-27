import json
import re
from SmartBite.chatbot.chains.base import ChainBase

class NutritionGoalSortingChain(ChainBase):
    """
    Chain to help users find recipes that align with specific nutritional goals.
    """

    def __init__(self, llm, recipe_file_path):
        """
        Initialize the NutritionGoalSortingChain with an LLM and a recipe JSON file.

        Args:
            llm (ChatOpenAI): The language model for processing queries.
            recipe_file_path (str): Path to the JSON file containing recipe data.
        """

        super().__init__(
            system_template="""
            You are a culinary assistant specializing in sorting recipes based on nutritional goals.
            Filter recipes from the provided database and rank them according to the user's specified nutritional goal.

            Recipe Database:
            {recipe_database}

            Ensure your response includes:
            - Up to 5 recipes that best match the user's request.
            - Present the recipes as a simple bulleted list with their relevance scores.
            - If no recipes match, provide a clear message indicating no results.
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

    def parse_input(self, user_input: str):
        """
        Parse user input to extract nutritional goals, categories, and optional values.

        Args:
            user_input (str): The user's input string specifying the nutritional goal.

        Returns:
            tuple: A tuple containing the goal (str), category (str), and value (float or None).
        """

        user_input = user_input.strip().lower()

        # Map user-friendly terms to database categories
        category_map = {
            "calorie": "energy",
            "protein": "protein",
            "fat": "fat",
            "sugar": "sugars",
            "fiber": "fibre",
            "salt": "salt",
            "carbohydrate": "carbs",
        }

        # Check for goal keywords
        if "high" in user_input or "more than" in user_input:
            goal = "high"
        elif "low" in user_input or "less than" in user_input:
            goal = "low"
        else:
            goal = None

        # Extract category
        category = next((cat for cat in category_map if cat in user_input), None)
        if category:
            category = category_map[category]  # Map to database field

        # Extract numeric value if present
        value = None
        match = re.search(r"(\d+)", user_input)
        if match:
            value = float(match.group(1))

        print(f"Parsed Input -> Goal: {goal}, Category: {category}, Value: {value}")
        return goal, category, value

    def process_input(self, user_input: str, max_recipes=5):
        """
        Prepare input data by filtering recipes based on nutritional goals.

        Args:
            user_input (str): User's input describing the nutritional goal.
            max_recipes (int): The maximum number of recipes to return. Defaults to 5.

        Returns:
            dict or None: A dictionary containing filtered recipes, user input, goal, and category, 
                        or None if no matching recipes are found.
        """

        goal, category, value = self.parse_input(user_input)

        if not goal or not category:
            return None  # Invalid input

        filtered_recipes = []

        for recipe in self.recipe_database:
            nutrition = {
                item["category"].lower(): float(item["value"])
                for item in recipe.get("nutrition_info", [])
            }

            if category not in nutrition:
                continue  # Skip recipes missing the requested category

            # Apply the filter based on the goal
            match = False
            if goal == "high" or goal == "more than":
                match = nutrition[category] > (value or 0)
            elif goal == "low" or goal == "less than":
                match = nutrition[category] < (value or float("inf"))

            if match:
                recipe["relevance"] = nutrition[category]  # Add relevance score for sorting
                filtered_recipes.append(recipe)

        # Sort recipes by relevance
        filtered_recipes.sort(key=lambda x: x["relevance"], reverse=(goal == "high"))

        # Limit the number of results
        limited_recipes = filtered_recipes[:max_recipes]

        if limited_recipes:
            return {
                "recipe_database": json.dumps(limited_recipes, indent=2),
                "user_input": user_input,
                "filtered_recipes": limited_recipes,
                "goal": goal.capitalize(),
                "category": category.capitalize()
            }
        else:
            return None

    def format_response(self, filtered_recipes, goal, category):
        """
        Format the response with the recipe titles and relevance scores.

        Args:
            filtered_recipes (list[dict]): List of filtered recipes with relevance scores.
            goal (str): The user's nutritional goal (e.g., "High", "Low").
            category (str): The nutritional category of interest (e.g., "Protein").

        Returns:
            str: A formatted string containing the recipe titles and relevance scores.
        """

        formatted_response = f"### Recipes with {goal} {category} Goals\n"
        for recipe in filtered_recipes:
            formatted_response += f"- {recipe['recipe_title']}: {recipe['relevance']}\n"
        return formatted_response

    def run(self, user_input: str):
        """
        Run the chain to get recipes filtered by nutritional goals.

        Args:
            user_input (str): The user's input specifying the nutritional goal.

        Returns:
            str: A response containing recipes matching the user's nutritional goal 
                or an error message if no matches are found.
        """

        inputs = self.process_input(user_input)

        if not inputs:
            return "Sorry, I couldn't find recipes matching the specified nutritional goal."

        # Structure messages for the LLM
        messages = [
            {"role": "system", "content": inputs["recipe_database"]},
            {"role": "user", "content": inputs["user_input"]}
        ]

        # Call the ChatOpenAI model and process the response
        response = self.llm.invoke(messages)

        # Validate and format the response
        filtered_recipes = inputs["filtered_recipes"]
        goal = inputs["goal"]
        category = inputs["category"]
        return self.format_response(filtered_recipes, goal, category)
