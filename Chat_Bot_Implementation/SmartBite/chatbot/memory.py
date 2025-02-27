import os
import json
from typing import List, Dict, Tuple
from datetime import datetime, timezone

class MemoryManager:
    """
    Manages session history for user interactions.

    This class provides support for in-memory storage and optional persistence to disk, allowing
    conversational history to be stored, retrieved, and managed efficiently.

    Attributes:
        max_messages (int): Maximum number of messages to retain in history.
        history_dir (str): Directory path for saving session history files.
        store (dict): In-memory storage for current session histories.
    """


    def __init__(self, max_messages: int = 50, history_dir: str = "session_history"):
        """
        Initialize the memory manager.

        Args:
            max_messages: Maximum number of messages to store in the history.
            history_dir: Directory to save session histories.
        """
        self.max_messages = max_messages
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)

        # In-memory store for current session histories
        self.store: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    def _get_history_file_path(self, user_id: str, conversation_id: str) -> str:
        """
        Generate the file path for a specific user's conversation history.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.

        Returns:
            str: The file path for the user's conversation history.
        """

        return os.path.join(self.history_dir, f"{user_id}_{conversation_id}_history.json")

    def load_session_history(self, user_id: str, conversation_id: str) -> None:
        """
        Load session history from a file into memory.

        If the session history file exists, it loads the history into the in-memory store.
        Otherwise, initializes an empty history for the user and conversation.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.
        """

        file_path = self._get_history_file_path(user_id, conversation_id)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                self.store[(user_id, conversation_id)] = json.load(file)
        else:
            self.store[(user_id, conversation_id)] = []

    def save_session_history(self, user_id: str, conversation_id: str) -> None:
        """
        Save the current session history to a file.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.
        """

        file_path = self._get_history_file_path(user_id, conversation_id)
        with open(file_path, "w") as file:
            json.dump(self.store.get((user_id, conversation_id), []), file, indent=2)

    def get_session_history(self, user_id: str, conversation_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the session history for a specific user and conversation.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.

        Returns:
            list[dict]: A list of message dictionaries, each containing:
                - role (str): The role of the message sender (e.g., "user" or "bot").
                - content (str): The content of the message.
                - timestamp (str): ISO-formatted timestamp of the message.
        """

        if (user_id, conversation_id) not in self.store:
            self.load_session_history(user_id, conversation_id)
        return self.store[(user_id, conversation_id)]

    def add_message_to_history(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        """
        Add a message to the session history.

        If the session does not exist, it initializes a new session and appends the message.
        Trims the history to the maximum number of messages if it exceeds the limit.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.
            role (str): The role of the message sender (e.g., "user" or "bot").
            content (str): The content of the message.
        """

        if (user_id, conversation_id) not in self.store:
            self.load_session_history(user_id, conversation_id)

        # Append the new message with a timestamp
        self.store[(user_id, conversation_id)].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Trim history to the maximum number of messages
        self.store[(user_id, conversation_id)] = self.store[(user_id, conversation_id)][-self.max_messages:]

    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """
        Clear the session history for a specific user and conversation.

        Removes the in-memory session history and deletes the corresponding history file, if it exists.

        Args:
            user_id (str): Identifier for the user.
            conversation_id (str): Identifier for the conversation.
        """

        self.store[(user_id, conversation_id)] = []
        file_path = self._get_history_file_path(user_id, conversation_id)
        if os.path.exists(file_path):
            os.remove(file_path)