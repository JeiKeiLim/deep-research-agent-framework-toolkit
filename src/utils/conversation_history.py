"""Conversation history management utilities.

Tools to manage conversation history for the Deep Research Agent.
Enables referencing past conversations and maintaining context
across sessions.

Author: Rijoo Kim
Contact: gureme1121@gmail.com
"""

import json
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------
# Domain Models (Pydantic v2)
# ---------------------------


class MessageRole(str, Enum):
    """Enumeration for message roles in conversations."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Individual message in a conversation."""

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        str_strip_whitespace=True,  # basic trimming
    )

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_non_empty(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("content must be a non-empty string")
        return v


class Conversation(BaseModel):
    """A complete conversation session."""

    schema_version: int = 1

    id: str
    title: str
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def title_non_empty(cls, v: str) -> str:
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError("title must be a non-empty string")
        return v

    def add_message(
        self,
        role: MessageRole | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new message to the conversation."""
        if isinstance(role, str):
            try:
                role = MessageRole(role)
            except ValueError as err:
                allowed = (
                    f"'{MessageRole.USER.value}' or '{MessageRole.ASSISTANT.value}'"
                )
                raise ValueError(f"Role must be {allowed}, got '{role}'") from err

        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )
        self.updated_at = datetime.utcnow()

    def get_context_summary(
        self,
        max_messages: int = 5,
        max_chars: int | None = None,
        ellipsis: str = "...",
    ) -> str:
        """Get a summary of recent conversation context for agent reference."""
        if not self.messages:
            return "No previous conversation context."

        recent = self.messages[-max_messages:]
        lines: list[str] = []
        for msg in recent:
            role_emoji = "👤" if msg.role == MessageRole.USER else "🤖"
            text = msg.content
            if isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars] + ellipsis
            lines.append(f"{role_emoji} {msg.role.value}: {text}")
        return "\n".join(lines)

    def get_searchable_content(self) -> str:
        """Get all content for search purposes."""
        return " ".join(m.content for m in self.messages)


class ConversationHistory:
    """Manages conversation history storage and retrieval."""

    def __init__(self, storage_dir: str = "conversation_history") -> None:
        """Initialize conversation history manager.

        Args:
            storage_dir: Directory to store conversation history files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.conversations: dict[str, Conversation] = {}
        self._load_conversations()

    def _load_conversations(self) -> None:
        """Load existing conversations from storage (legacy-safe)."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    raw = f.read()
                # Use model_validate_json to handle legacy JSON format
                conv = Conversation.model_validate_json(raw)
                self.conversations[conv.id] = conv
            except Exception as e:
                print(f"Error loading conversation from {file_path}: {e}")

    def _save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to storage."""
        file_path = self.storage_dir / f"{conversation.id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Serialization with model_dump(mode="json")
                # ensures compatibility including ISO8601 datetime format
                json.dump(
                    conversation.model_dump(mode="json"),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"Error saving conversation to {file_path}: {e}")

    def create_conversation(
        self, title: str, initial_message: str | None = None
    ) -> str:
        """Create a new conversation.

        Args:
            title: Title for the conversation
            initial_message: Optional initial message content

        Returns:
        -------
            Conversation ID
        """
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Title must be a non-empty string")

        if initial_message is not None and (
            not isinstance(initial_message, str) or not initial_message.strip()
        ):
            raise ValueError("Initial message must be a non-empty string if provided")

        conversation_id = f"conv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.conversations)}"

        conv = Conversation(
            id=conversation_id,
            title=title,
        )

        if initial_message:
            conv.add_message(MessageRole.USER, initial_message)

        self.conversations[conversation_id] = conv
        self._save_conversation(conv)
        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a message to an existing conversation.

        Returns:
        -------
            True if successful, False if conversation not found
        """
        conv = self.conversations.get(conversation_id)
        if conv is None:
            return False

        conv.add_message(role, content, metadata)
        self._save_conversation(conv)
        return True

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """Update the title of an existing conversation."""
        conv = self.conversations.get(conversation_id)
        if conv is None:
            return False
        conv.title = new_title  # Blocks empty titles from pydantic validation
        conv.updated_at = datetime.utcnow()
        self._save_conversation(conv)
        return True

    def get_all_conversations(self) -> list[Conversation]:
        """Get all conversations."""
        return list(self.conversations.values())

    def search_conversations(
        self, query: str, max_results: int = 10
    ) -> list[Conversation]:
        """Search conversations by content."""
        q = query.lower()
        matched: list[Conversation] = []
        for conv in self.conversations.values():
            if q in conv.title.lower() or q in conv.get_searchable_content().lower():
                matched.append(conv)
        matched.sort(key=lambda c: c.updated_at, reverse=True)
        return matched[:max_results]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id not in self.conversations:
            return False

        del self.conversations[conversation_id]

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
        """Get conversation context for agent reference."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return "No conversation context available."
        return conv.get_context_summary(max_messages)


class ConversationManager:
    """Manages conversation lifecycle and provides enhanced conversation context.

    This class handles:
    - Starting new conversations
    - Managing conversation context
    - Handling title updates with callbacks
    - Providing enhanced queries with conversation context
    """

    def __init__(self, history: ConversationHistory, max_messages: int = 5) -> None:
        """Initializes the ConversationManager.

        Args:
            history: An instance of ConversationHistory to manage.
            max_messages: The maximum number of messages to keep in context.
        """
        self.history = history
        self.max_messages = max_messages
        self.current_conversation_id: str | None = None
        self.title_update_callbacks: list[Callable[[str, str], None]] = []

    def add_title_update_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add a callback function to be called when conversation title is updated."""
        self.title_update_callbacks.append(callback)

    def remove_title_update_callback(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Remove a title update callback function."""
        if callback in self.title_update_callbacks:
            self.title_update_callbacks.remove(callback)

    def start_new_conversation(self, title: str | None = None) -> str:
        """Start a new conversation with optional title."""
        if title is None:
            title = f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
        self.current_conversation_id = self.history.create_conversation(title)
        return self.current_conversation_id

    def get_enhanced_query(self, query: str, max_messages: int = 10) -> str:
        """Get an enhanced query with conversation context."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be a positive integer")

        if not self.current_conversation_id:
            self.current_conversation_id = self.start_new_conversation()

        # Context before adding new message
        context = self.history.get_conversation_context(
            self.current_conversation_id, max_messages
        )

        # If the current conversation has no messages, set the title to the query
        current_conv = self.history.get_conversation(self.current_conversation_id)
        if current_conv and len(current_conv.messages) == 0:
            self.history.update_conversation_title(self.current_conversation_id, query)
            for cb in list(self.title_update_callbacks):
                try:
                    cb(self.current_conversation_id, query)
                except Exception as e:
                    print(f"Error in title update callback: {e}")

        # Append user message
        self.history.add_message(self.current_conversation_id, MessageRole.USER, query)

        if context and context != "No conversation context available.":
            return (
                f"Previous conversation context:\n"
                f"{context}\n\n"
                f"Current question: {query}\n\n"
                f"Please answer by referring to the previous conversation above, "
                f"and maintain continuity with it in your response."
            )
        return query

    def add_assistant_message(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Add an assistant message to the current conversation."""
        if not self.current_conversation_id:
            return False
        return self.history.add_message(
            self.current_conversation_id, MessageRole.ASSISTANT, content, metadata
        )

    def get_conversation_context(self, max_messages: int | None = None) -> str:
        """Get conversation context for the current conversation."""
        if not self.current_conversation_id:
            return "No conversation context available."
        if max_messages is None:
            max_messages = self.max_messages
        return self.history.get_conversation_context(
            self.current_conversation_id, max_messages
        )

    def search_history(self, query: str, max_results: int = 10) -> list[Conversation]:
        """Search conversations in history."""
        return self.history.search_conversations(query, max_results)

    def get_all_conversations(self) -> list[Conversation]:
        """Get all conversations from history."""
        return self.history.get_all_conversations()

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation."""
        if conversation_id in self.history.conversations:
            self.current_conversation_id = conversation_id
            return True
        return False

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a specific conversation by ID."""
        return self.history.get_conversation(conversation_id)

    def get_current_conversation_id(self) -> str | None:
        """Get the current conversation ID."""
        return self.current_conversation_id

    def is_conversation_active(self) -> bool:
        """Check if there is an active conversation."""
        return self.current_conversation_id is not None

    def get_conversation_title(self) -> str | None:
        """Get the title of the current conversation."""
        if not self.current_conversation_id:
            return None
        conv = self.history.get_conversation(self.current_conversation_id)
        return conv.title if conv else None

    def get_conversation_message_count(self) -> int:
        """Get the number of messages in the current conversation."""
        if not self.current_conversation_id:
            return 0
        conv = self.history.get_conversation(self.current_conversation_id)
        return len(conv.messages) if conv else 0
