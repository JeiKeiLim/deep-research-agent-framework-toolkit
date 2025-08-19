"""Conversation History Management.

Tools to manage conversation history for the Deep Research Agent.
Enables referencing past conversations and maintaining context
across sessions.

Author: Rijoo Kim
Contact: gureme1121@gmail.com
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Message:
    """Individual message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """A complete conversation session."""

    id: str
    title: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            messages=[Message.from_dict(msg) for msg in data["messages"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new message to the conversation."""
        message = Message(
            role=role, content=content, timestamp=datetime.now(), metadata=metadata
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_context_summary(self, max_messages: int = 5) -> str:
        """Get a summary of recent conversation context for agent reference."""
        if not self.messages:
            return "No previous conversation context."

        recent_messages = self.messages[-max_messages:]
        context_lines = []

        for msg in recent_messages:
            role_emoji = "👤" if msg.role == "user" else "🤖"
            context_lines.append(
                f"{role_emoji} {msg.role}: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}"
            )

        return "\n".join(context_lines)

    def get_searchable_content(self) -> str:
        """Get all content for search purposes."""
        return " ".join([msg.content for msg in self.messages])


class ConversationHistory:
    """Manages conversation history storage and retrieval."""

    def __init__(self, storage_dir: str = "conversation_history"):
        """Initialize conversation history manager.

        Args:
            storage_dir: Directory to store conversation history files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.conversations: Dict[str, Conversation] = {}
        self._load_conversations()

    def _load_conversations(self) -> None:
        """Load existing conversations from storage."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversation = Conversation.from_dict(data)
                    self.conversations[conversation.id] = conversation
            except Exception as e:
                print(f"Error loading conversation from {file_path}: {e}")

    def _save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to storage."""
        file_path = self.storage_dir / f"{conversation.id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversation to {file_path}: {e}")

    def create_conversation(
        self, title: str, initial_message: Optional[str] = None
    ) -> str:
        """Create a new conversation.

        Args:
            title: Title for the conversation
            initial_message: Optional initial message content

        Returns
        -------
            Conversation ID
        """
        conversation_id = (
            f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.conversations)}"
        )

        conversation = Conversation(
            id=conversation_id,
            title=title,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        if initial_message:
            conversation.add_message("user", initial_message)

        self.conversations[conversation_id] = conversation
        self._save_conversation(conversation)

        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a message to an existing conversation.

        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender ("user" or "assistant")
            content: Message content
            metadata: Optional metadata

        Returns
        -------
            True if successful, False if conversation not found
        """
        if conversation_id not in self.conversations:
            return False

        conversation = self.conversations[conversation_id]
        conversation.add_message(role, content, metadata)
        self._save_conversation(conversation)
        return True

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation

        Returns
        -------
            Conversation object or None if not found
        """
        return self.conversations.get(conversation_id)

    def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """Update the title of an existing conversation.

        Args:
            conversation_id: ID of the conversation
            new_title: New title for the conversation

        Returns
        -------
            True if successful, False if conversation not found
        """
        if conversation_id not in self.conversations:
            return False

        conversation = self.conversations[conversation_id]
        conversation.title = new_title
        conversation.updated_at = datetime.now()
        self._save_conversation(conversation)
        return True

    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations.

        Returns
        -------
            List of all conversations
        """
        return list(self.conversations.values())

    def search_conversations(
        self, query: str, max_results: int = 10
    ) -> List[Conversation]:
        """Search conversations by content.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns
        -------
            List of matching conversations
        """
        query_lower = query.lower()
        matching_conversations = []

        for conversation in self.conversations.values():
            if (
                query_lower in conversation.title.lower()
                or query_lower in conversation.get_searchable_content().lower()
            ):
                matching_conversations.append(conversation)

        # Sort by most recent first
        matching_conversations.sort(key=lambda x: x.updated_at, reverse=True)

        return matching_conversations[:max_results]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: ID of the conversation to delete

        Returns
        -------
            True if successful, False if conversation not found
        """
        if conversation_id not in self.conversations:
            return False

        # Remove from memory
        del self.conversations[conversation_id]

        # Remove from storage
        file_path = self.storage_dir / f"{conversation_id}.json"
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error deleting conversation file {file_path}: {e}")

        return True

    def get_conversation_context(
        self, conversation_id: str, max_messages: int = 5
    ) -> str:
        """Get conversation context for agent reference.

        Args:
            conversation_id: ID of the conversation
            max_messages: Maximum number of recent messages to include

        Returns
        -------
            Context string for the agent
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return "No conversation context available."

        return conversation.get_context_summary(max_messages)


class ConversationManager:
    """Manages conversation lifecycle and provides enhanced conversation context.
    
    This class handles the complete conversation management including:
    - Starting new conversations
    - Managing conversation context
    - Handling title updates with callbacks
    - Providing enhanced queries with conversation context
    """

    def __init__(self, history: ConversationHistory, max_messages: int = 5):
        """Initialize conversation manager.
        
        Args:
            history: ConversationHistory instance for storage operations
            max_messages: Default maximum number of recent messages to include in context
        """
        self.history = history
        self.max_messages = max_messages
        self.current_conversation_id: Optional[str] = None
        self.title_update_callbacks: List[Callable[[str, str], None]] = []

    def add_title_update_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add a title update callback.
        
        Args:
            callback: A callable that takes (conversation_id, new_title) as arguments.
        """
        self.title_update_callbacks.append(callback)

    def remove_title_update_callback(self, callback: Callable[[str, str], None]) -> None:
        """Remove a title update callback.
        
        Args:
            callback: A callable that was previously registered.
        """
        if callback in self.title_update_callbacks:
            self.title_update_callbacks.remove(callback)

    def start_new_conversation(self, title: str = None) -> str:
        """Start a new conversation session.
        
        Args:
            title: Optional title for the conversation. If None, will use timestamp.
            
        Returns:
            Conversation ID
        """
        if title is None:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        self.current_conversation_id = self.history.create_conversation(title)
        return self.current_conversation_id

    def get_enhanced_query(self, query: str, max_messages: int = 10) -> str:
        """Get enhanced query with conversation context and manage conversation lifecycle.
        
        This method handles:
        1. Starting new conversation if none exists
        2. Setting conversation title for first message
        3. Adding user message to history
        4. Returning enhanced query with conversation context
        
        Args:
            query: The user's query
            max_messages: Maximum number of recent messages to include in context
            
        Returns:
            Enhanced query with conversation context
        """
        # Start new conversation if none exists
        if not self.current_conversation_id:
            self.start_new_conversation()

        # Get conversation context for agent reference
        conversation_context = self.history.get_conversation_context(
            self.current_conversation_id, max_messages
        )

        # Check if this is the first user message (set as title)
        current_conv = self.history.get_conversation(self.current_conversation_id)
        if current_conv and len(current_conv.messages) == 0:
            # First message - set as conversation title
            self.history.update_conversation_title(
                self.current_conversation_id, query
            )
            print(f"Set conversation title to: {query}")

            # Notify title update callbacks
            for callback in self.title_update_callbacks:
                try:
                    callback(self.current_conversation_id, query)
                except Exception as e:
                    print(f"Error in title update callback: {e}")

        # Add user message to history
        self.history.add_message(self.current_conversation_id, "user", query)

        # Return enhanced query with conversation context
        if (
            conversation_context
            and conversation_context != "No conversation context available."
        ):
            return f"""Previous conversation context:
            {conversation_context}

            Current question: {query}

            Please answer by referring to the previous conversation above, and maintain continuity with it in your response."""
        
        return query

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add assistant response to conversation history.
        
        Args:
            content: The assistant's response content
            metadata: Optional metadata
            
        Returns:
            True if successful, False if no current conversation
        """
        if not self.current_conversation_id:
            return False
            
        return self.history.add_message(
            self.current_conversation_id, "assistant", content, metadata
        )

    def get_conversation_context(self, max_messages: int = None) -> str:
        """Get current conversation context for agent reference.
        
        Args:
            max_messages: Maximum number of recent messages to include.
                         If None, uses the configured default value.
            
        Returns:
            Context string for the agent
        """
        if not self.current_conversation_id:
            return "No conversation context available."

        # Use configured default if max_messages is not specified
        if max_messages is None:
            max_messages = self.max_messages

        return self.history.get_conversation_context(
            self.current_conversation_id, max_messages
        )

    def search_history(self, query: str, max_results: int = 10) -> List[Conversation]:
        """Search conversation history.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching conversations
        """
        return self.history.search_conversations(query, max_results)

    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations.
        
        Returns:
            List of all conversations
        """
        return self.history.get_all_conversations()

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation.
        
        Args:
            conversation_id: ID of the conversation to switch to
            
        Returns:
            True if successful, False if conversation not found
        """
        if conversation_id in self.history.conversations:
            self.current_conversation_id = conversation_id
            return True
        return False

    def get_conversation(self, conversation_id: str):
        """Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation object or None if not found
        """
        return self.history.get_conversation(conversation_id)

    # Convenience methods for external access
    def get_current_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID.
        
        Returns:
            Current conversation ID or None if no conversation is active
        """
        return self.current_conversation_id

    def is_conversation_active(self) -> bool:
        """Check if there's an active conversation.
        
        Returns:
            True if there's an active conversation, False otherwise
        """
        return self.current_conversation_id is not None

    def get_conversation_title(self) -> Optional[str]:
        """Get the title of the current conversation.
        
        Returns:
            Conversation title or None if no conversation is active
        """
        if not self.current_conversation_id:
            return None
            
        conversation = self.history.get_conversation(self.current_conversation_id)
        return conversation.title if conversation else None

    def get_conversation_message_count(self) -> int:
        """Get the number of messages in the current conversation.
        
        Returns:
            Number of messages in the current conversation, 0 if no conversation
        """
        if not self.current_conversation_id:
            return 0
            
        conversation = self.history.get_conversation(self.current_conversation_id)
        return len(conversation.messages) if conversation else 0
