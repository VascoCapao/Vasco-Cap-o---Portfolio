import sqlite3
import re

class SaveFavoriteRecipeChain:
    def __init__(self, db_name="SmartBite/chatbot/data/Database/recipes.db"):
        """
        Initialize the SaveFavoriteRecipeChain with the database file.

        Args:
            db_name (str): Path to the SQLite database file. Defaults to 
                        "SmartBite/chatbot/data/Database/recipes.db".
        """

        self.db_name = db_name

    def execute_query(self, query, params=()):
        """
        Execute an SQL query on the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Parameters for the SQL query. Defaults to an empty tuple.

        Returns:
            list or str: Query results as a list of tuples, or an error message if an exception occurs.
        """

        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            return f"Database error: {e}"
        finally:
            conn.close()

    def find_recipe_by_title(self, title):
        """
        Find a recipe by its title.

        Args:
            title (str): The title of the recipe to search for.

        Returns:
            list[tuple]: A list containing the recipe ID and title if found, or an empty list if no match is found.
        """

        query = """
        SELECT recipe_id, title
        FROM recipes
        WHERE title LIKE ?
        LIMIT 1
        """
        return self.execute_query(query, (f"%{title}%",))

    def save_to_favorites(self, user_id, recipe_title):
        """
        Save a recipe to the user's favorites.

        Args:
            user_id (int): ID of the user.
            recipe_title (str): Title of the recipe to save.

        Returns:
            str: A message indicating whether the recipe was successfully added, already in favorites, or not found.
        """

        matching_recipes = self.find_recipe_by_title(recipe_title)
        if not matching_recipes:
            return f"No recipes found matching '{recipe_title}'."

        recipe_id, title = matching_recipes[0]
        query_check = "SELECT favorite_id FROM favorite_recipe WHERE user_id = ? AND recipe_id = ?"
        if self.execute_query(query_check, (user_id, recipe_id)):
            return f"Recipe '{title}' is already in your favorites."

        query_insert = """
        INSERT INTO favorite_recipe (user_id, recipe_id)
        VALUES (?, ?)
        """
        self.execute_query(query_insert, (user_id, recipe_id))
        return f"Recipe '{title}' has been added to your favorites."

    def remove_from_favorites(self, user_id, recipe_title):
        """
        Remove a recipe from the user's favorites.

        Args:
            user_id (int): ID of the user.
            recipe_title (str): Title of the recipe to remove.

        Returns:
            str: A message indicating whether the recipe was successfully removed, not in favorites, or not found.
        """

        matching_recipes = self.find_recipe_by_title(recipe_title)
        if not matching_recipes:
            return f"No recipes found matching '{recipe_title}'."

        recipe_id, title = matching_recipes[0]
        query_check = "SELECT favorite_id FROM favorite_recipe WHERE user_id = ? AND recipe_id = ?"
        if not self.execute_query(query_check, (user_id, recipe_id)):
            return f"Recipe '{title}' is not in your favorites."

        query_delete = """
        DELETE FROM favorite_recipe
        WHERE user_id = ? AND recipe_id = ?
        """
        self.execute_query(query_delete, (user_id, recipe_id))
        return f"Recipe '{title}' has been removed from your favorites."

    def run(self, user_input, user_id):
        """
        Process user input to add or remove a recipe from favorites.

        Args:
            user_input (str): User's input specifying the action and recipe title.
            user_id (int): ID of the user making the request.

        Returns:
            str: A message indicating the result of the action or an error message if input is not recognized.
        """

        if "remove" in user_input.lower():
            match = re.search(r"remove (.+) recipe from favorites", user_input, re.IGNORECASE)
            if match:
                return self.remove_from_favorites(user_id, match.group(1).strip())
            else:
                return "Please specify which recipe you want to remove from favorites."
        elif "save" in user_input.lower():
            match = re.search(r"save (.+) recipe to favorites", user_input, re.IGNORECASE)
            if match:
                return self.save_to_favorites(user_id, match.group(1).strip())
            else:
                return "Please specify which recipe you want to add to favorites."
        else:
            return "Input not recognized. Please specify 'add' or 'remove' with a recipe title."