#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract Base Class for Vision-Language Models (VLM)

This module defines the abstract interface that all VLM models should implement
to ensure compatibility with the Gradio UI interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Tuple


class VLMModelBase(ABC):
    """
    Abstract base class for Vision-Language Models.
    
    This class defines the interface that all VLM models must implement
    to work with the Gradio chat interface.
    """
    
    def __init__(self):
        """Initialize the VLM model."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and processor.
        
        This method should initialize all necessary components
        for the model to function properly.
        """
        pass
    
    @abstractmethod
    def validate_files(self, files: List[Any]) -> Tuple[bool, str]:
        """
        Validate uploaded files according to model constraints.
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        pass
    
    @abstractmethod
    def process_files(self, files: List[Any]) -> List[Dict[str, Any]]:
        """
        Process uploaded files and convert them to model-compatible format.
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            List of content objects for model input
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self, 
        conversation_history: List[Dict[str, Any]], 
        system_prompt: str
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from the model.
        
        Args:
            conversation_history: List of conversation messages
            system_prompt: System prompt for the conversation
            
        Yields:
            Formatted text fragments during generation
        """
        pass
    
    @abstractmethod
    def stop_generation(self) -> None:
        """
        Stop the current generation process.
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.__class__.__name__,
            "description": "Vision-Language Model",
            "supported_formats": ["text", "image", "video"]
        }
    
    def format_message_content(self, content: Any) -> str:
        """
        Format message content for display in the chat interface.
        
        Args:
            content: Message content (string or list of content objects)
            
        Returns:
            Formatted content string for display
        """
        if isinstance(content, list):
            text_parts = []
            file_count = 0
            
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item["text"])
                else:
                    file_count += 1
            
            display_text = " ".join(text_parts)
            
            if file_count > 0:
                return f"[{file_count} file(s) uploaded]\n{display_text}"
            return display_text
        
        return str(content)