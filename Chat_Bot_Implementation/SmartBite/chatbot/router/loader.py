import os
from semantic_router import RouteLayer

# Adjust to the exact path of your layer.json file
FILENAME = "SmartBite/chatbot/router/layer.json"

def load_intention_classifier() -> RouteLayer:
    """
    Load the `layer.json` file in the `router` folder.
    Returns:
        RouteLayer object to classify user intentions.
    Raises:
        FileNotFoundError: If the file is not found.
    """
    # Ensure the file exists before loading
    if not os.path.exists(FILENAME):
        raise FileNotFoundError(f"File not found: {FILENAME}")

    # Load the RouteLayer from the JSON file
    rl = RouteLayer.from_json(FILENAME)
    return rl

