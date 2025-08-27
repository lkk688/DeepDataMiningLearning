#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio UI Module for VLM Chat Interface

This module provides a clean, reusable Gradio interface that can work with
any Vision-Language Model that implements the VLMModelBase interface.
"""

import copy
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr

from vlm_base import VLMModelBase


class VLMGradioInterface:
    """
    A reusable Gradio interface for Vision-Language Models.
    
    This class creates a chat interface that can work with any VLM model
    that implements the VLMModelBase abstract interface.
    """
    
    def __init__(self, model: VLMModelBase, title: str = "VLM Chat Interface"):
        """
        Initialize the Gradio interface.
        
        Args:
            model: VLM model instance that implements VLMModelBase
            title: Title for the Gradio interface
        """
        self.model = model
        self.title = title
        self.conversation_history = []  # Store raw conversation history
        
        # CSS styling for the interface
        self.css = """
        .chatbot {
            height: 600px;
        }
        .upload-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
        .system-prompt {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        """
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and return the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(css=self.css, title=self.title) as interface:
            # Header
            gr.Markdown(f"# {self.title}")
            
            model_info = self.model.get_model_info()
            gr.Markdown(
                f"**Model:** {model_info.get('name', 'Unknown')}\n\n"
                f"**Description:** {model_info.get('description', 'No description available')}\n\n"
                f"**Supported Formats:** {', '.join(model_info.get('supported_formats', []))}"
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        elem_classes=["chatbot"],
                        show_copy_button=True,
                        bubble_full_width=False
                    )
                    
                    # Message input
                    with gr.Row():
                        message_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=3,
                            scale=4
                        )
                        send_button = gr.Button("Send", variant="primary", scale=1)
                    
                    # File upload
                    file_upload = gr.File(
                        label="Upload Files (Images, Videos, PDFs, PPTs)",
                        file_count="multiple",
                        elem_classes=["upload-container"]
                    )
                    
                    # Control buttons
                    with gr.Row():
                        clear_button = gr.Button("Clear Conversation", variant="secondary")
                        stop_button = gr.Button("Stop Generation", variant="stop")
                
                with gr.Column(scale=1):
                    # System prompt
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder="Enter system prompt (optional)...",
                        lines=5,
                        elem_classes=["system-prompt"]
                    )
                    
                    # Model information display
                    gr.Markdown("### Model Constraints")
                    constraints = self._get_model_constraints()
                    gr.Markdown(constraints)
            
            # Event handlers using modern Gradio API
            chat_inputs = [message_input, file_upload, system_prompt, chatbot]
            chat_outputs = [message_input, file_upload, chatbot]
            
            send_button.click(
                fn=self._handle_chat,
                inputs=chat_inputs,
                outputs=chat_outputs,
                show_progress=True
            )
            
            message_input.submit(
                fn=self._handle_chat,
                inputs=chat_inputs,
                outputs=chat_outputs,
                show_progress=True
            )
            
            clear_button.click(
                fn=self._clear_conversation,
                outputs=[chatbot, message_input, file_upload]
            )
            
            stop_button.click(
                fn=self._stop_generation
            )
        
        return interface
    
    def _handle_chat(
        self, 
        message: str, 
        files: List[Any], 
        system_prompt: str, 
        chat_history: List[List[str]]
    ) -> Tuple[str, None, List[List[str]]]:
        """
        Handle chat message and generate response.
        
        Args:
            message: User message text
            files: Uploaded files
            system_prompt: System prompt text
            chat_history: Current chat history for display
            
        Returns:
            Tuple of (cleared_message, cleared_files, updated_chat_history)
        """
        if not message.strip() and not files:
            return message, files, chat_history
        
        # Validate files
        is_valid, error_msg = self.model.validate_files(files)
        if not is_valid:
            # Show error in chat
            error_history = chat_history + [[message, f"❌ Error: {error_msg}"]]
            return "", None, error_history
        
        # Process files and create user message content
        file_content = self.model.process_files(files)
        user_content = []
        
        # Add text content if provided
        if message.strip():
            user_content.append({"type": "text", "text": message.strip()})
        
        # Add file content
        user_content.extend(file_content)
        
        # Add user message to conversation history
        user_message = {"role": "user", "content": user_content}
        self.conversation_history.append(user_message)
        
        # Format user message for display
        display_message = self.model.format_message_content(user_content)
        updated_history = chat_history + [[display_message, ""]]
        
        # Generate response
        try:
            response_text = ""
            for chunk in self.model.stream_generate(self.conversation_history, system_prompt):
                response_text = chunk
                # Update the last message in chat history with streaming response
                updated_history[-1][1] = response_text
                yield "", None, updated_history
            
            # Add assistant response to conversation history
            assistant_message = {"role": "assistant", "content": response_text}
            self.conversation_history.append(assistant_message)
            
        except Exception as e:
            error_response = f"❌ Error generating response: {str(e)}"
            updated_history[-1][1] = error_response
            
            # Add error to conversation history
            assistant_message = {"role": "assistant", "content": error_response}
            self.conversation_history.append(assistant_message)
        
        return "", None, updated_history
    
    def _clear_conversation(self) -> Tuple[List, str, None]:
        """
        Clear the conversation history.
        
        Returns:
            Tuple of (empty_chat_history, empty_message, no_files)
        """
        self.conversation_history = []
        return [], "", None
    
    def _stop_generation(self) -> None:
        """
        Stop the current generation process.
        """
        self.model.stop_generation()
    
    def _get_model_constraints(self) -> str:
        """
        Get model constraints as formatted markdown.
        
        Returns:
            Formatted constraints string
        """
        model_info = self.model.get_model_info()
        
        constraints = []
        
        if "max_images" in model_info:
            constraints.append(f"• Max images: {model_info['max_images']}")
        
        if "max_videos" in model_info:
            constraints.append(f"• Max videos: {model_info['max_videos']}")
        
        if not constraints:
            constraints.append("• No specific constraints")
        
        return "\n".join(constraints)
    
    def launch(
        self, 
        server_name: str = "127.0.0.1", 
        server_port: int = 7860, 
        share: bool = False,
        **kwargs
    ) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            server_name: Server IP address
            server_port: Server port number
            share: Enable public sharing
            **kwargs: Additional arguments for Gradio launch
        """
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            **kwargs
        )


def create_vlm_interface(
    model: VLMModelBase, 
    title: str = "VLM Chat Interface"
) -> VLMGradioInterface:
    """
    Factory function to create a VLM Gradio interface.
    
    Args:
        model: VLM model instance that implements VLMModelBase
        title: Title for the interface
        
    Returns:
        Configured VLMGradioInterface instance
    """
    return VLMGradioInterface(model, title)