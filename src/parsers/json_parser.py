"""Parser for ChatGPT JSON exports."""

import json
import logging
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None


@dataclass
class Conversation:
    """Represents a complete conversation."""
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ChatGPTJSONParser:
    """Parser for ChatGPT conversations.json exports."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_export(self, file_path: Path) -> List[Conversation]:
        """Parse the complete ChatGPT export file.
        
        Args:
            file_path: Path to the conversations.json file
            
        Returns:
            List of parsed conversations
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Export file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = []
            for conv_data in data:
                conversation = self._parse_conversation(conv_data)
                if conversation:
                    conversations.append(conversation)
            
            self.logger.info(f"Parsed {len(conversations)} conversations from {file_path}")
            return conversations
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing export file: {e}")
    
    def _parse_conversation(self, conv_data: Dict) -> Optional[Conversation]:
        """Parse a single conversation from the export data.
        
        Args:
            conv_data: Dictionary containing conversation data
            
        Returns:
            Parsed Conversation object or None if invalid
        """
        try:
            conv_id = conv_data.get('id', '')
            title = conv_data.get('title', 'Untitled Conversation')
            
            # Parse timestamps
            created_at = self._parse_timestamp(conv_data.get('create_time'))
            updated_at = self._parse_timestamp(conv_data.get('update_time'))
            
            # Parse messages
            messages = []
            mapping = conv_data.get('mapping', {})
            
            for msg_id, msg_data in mapping.items():
                message = self._parse_message(msg_data, msg_id)
                if message:
                    messages.append(message)
            
            # Sort messages by timestamp if available
            messages.sort(key=lambda x: x.timestamp or datetime.min)
            
            return Conversation(
                id=conv_id,
                title=title,
                messages=messages,
                created_at=created_at,
                updated_at=updated_at
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse conversation: {e}")
            return None
    
    def _parse_message(self, msg_data: Dict, msg_id: str) -> Optional[ChatMessage]:
        """Parse a single message from the conversation mapping.
        
        Args:
            msg_data: Dictionary containing message data
            msg_id: Message ID
            
        Returns:
            Parsed ChatMessage object or None if invalid
        """
        try:
            message_info = msg_data.get('message')
            if not message_info:
                return None
            
            # Extract role and content
            author = message_info.get('author', {})
            role = author.get('role', 'unknown')
            
            # Skip system messages
            if role == 'system':
                return None
            
            content_parts = message_info.get('content', {}).get('parts', [])
            if not content_parts:
                return None
            
            # Join all content parts
            content = '\\n'.join(str(part) for part in content_parts if part)
            
            if not content.strip():
                return None
            
            # Parse timestamp
            timestamp = self._parse_timestamp(message_info.get('create_time'))
            
            return ChatMessage(
                role=role,
                content=content,
                timestamp=timestamp,
                message_id=msg_id
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse message {msg_id}: {e}")
            return None
    
    def _parse_timestamp(self, timestamp_value) -> Optional[datetime]:
        """Parse timestamp from various formats.
        
        Args:
            timestamp_value: Timestamp value (float, string, or None)
            
        Returns:
            Parsed datetime object or None
        """
        if not timestamp_value:
            return None
        
        try:
            if isinstance(timestamp_value, (int, float)):
                return datetime.fromtimestamp(timestamp_value)
            elif isinstance(timestamp_value, str):
                # Try parsing ISO format
                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            pass
        
        return None
    
    def get_conversation_chunks(self, conversation: Conversation, 
                             chunk_size: int = 1500, 
                             overlap: int = 200) -> Generator[str, None, None]:
        """Split conversation into chunks for processing.
        
        Args:
            conversation: Conversation to chunk
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Overlap between chunks in tokens
            
        Yields:
            Text chunks from the conversation
        """
        # Convert conversation to text
        text_parts = []
        text_parts.append(f"Title: {conversation.title}")
        
        for message in conversation.messages:
            text_parts.append(f"\\n{message.role.upper()}: {message.content}")
        
        full_text = '\\n'.join(text_parts)
        
        # Simple chunking by character count (approximate token estimation)
        # 1 token ≈ 4 characters on average
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        
        start = 0
        while start < len(full_text):
            end = min(start + char_chunk_size, len(full_text))
            
            # Try to break at a natural boundary
            if end < len(full_text):
                # Find the last newline or sentence boundary
                last_newline = full_text.rfind('\\n', start, end)
                last_period = full_text.rfind('.', start, end)
                
                if last_newline > start + char_chunk_size // 2:
                    end = last_newline
                elif last_period > start + char_chunk_size // 2:
                    end = last_period + 1
            
            chunk = full_text[start:end].strip()
            if chunk:
                yield chunk
            
            start = max(start + char_chunk_size - char_overlap, end)
            if start >= len(full_text):
                break