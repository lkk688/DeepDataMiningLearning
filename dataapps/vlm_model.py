#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-4V Model Implementation

This module contains the GLM-4V model implementation that extends the VLMModelBase
to provide specific functionality for GLM-4.1V-9B-Thinking and GLM-4.5V models.
"""

import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple

import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from vlm_base import VLMModelBase

# Optional imports with fallbacks
try:
    import fitz  # PyMuPDF for PDF processing
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF (fitz) not available. PDF processing will be disabled.")

try:
    import spaces  # Hugging Face Spaces GPU decorator
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Create a dummy decorator if spaces is not available
    def spaces_gpu_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    spaces = type('spaces', (), {'GPU': spaces_gpu_decorator})()

# Try to import specific GLM model classes, fall back to AutoModel
try:
    from transformers import GLM4VForConditionalGeneration
    HAS_GLM4V = True
except ImportError:
    HAS_GLM4V = False
    print("Warning: GLM4V model classes not found, using AutoModelForCausalLM")

try:
    from transformers import GLM4VMoeForConditionalGeneration
    HAS_GLM4V_MOE = True
except ImportError:
    HAS_GLM4V_MOE = False


class GLM4VModel(VLMModelBase):
    """
    GLM-4V model implementation for multimodal chat interface.
    
    This class provides methods for:
    - Processing multimodal inputs (text, images, videos, documents)
    - Converting documents (PDF, PPT) to images
    - Formatting model inputs and outputs
    - Streaming text generation with thinking process visualization
    """
    
    def __init__(self, model_path: str = "zai-org/GLM-4.1V-9B-Thinking"):
        """
        Initialize the GLM4VModel.
        
        Args:
            model_path: Hugging Face model identifier
        """
        super().__init__()
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.stop_generation_flag = False
    
    def load_model(self) -> None:
        """
        Initialize and load the GLM-4V model and processor.
        
        This function:
        1. Loads the appropriate processor for tokenization and input preparation
        2. Determines the correct model class based on the model path
        3. Loads the model with automatic device mapping and dtype selection
        
        The function supports both standard GLM-4V and MoE (Mixture of Experts) variants.
        """
        # Load the processor for handling multimodal inputs
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Select appropriate model class based on model name
        if "GLM-4.5V" in self.model_path:
            # Use MoE variant for GLM-4.5V models
            if HAS_GLM4V_MOE:
                self.model = GLM4VMoeForConditionalGeneration.from_pretrained(
                    self.model_path, 
                    torch_dtype="auto",    # Automatic dtype selection for efficiency
                    device_map="auto"      # Automatic device placement across GPUs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    torch_dtype="auto", 
                    device_map="auto",
                    trust_remote_code=True
                )
        else:
            # Use standard variant for other GLM-4V models
            if HAS_GLM4V:
                self.model = GLM4VForConditionalGeneration.from_pretrained(
                    self.model_path, 
                    torch_dtype="auto", 
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    torch_dtype="auto", 
                    device_map="auto",
                    trust_remote_code=True
                )
    
    def validate_files(self, files: List[Any]) -> Tuple[bool, str]:
        """
        Validate uploaded files according to GLM-4V model constraints.
        
        The GLM-4V model has specific limitations on multimodal inputs:
        - Maximum 10 images per conversation
        - Only 1 video, PPT, or PDF file allowed
        - Cannot mix different media types in the same message
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if not files:
            return True, ""
        
        # Count different file types
        vids = imgs = ppts = pdfs = 0
        
        for f in files:
            ext = Path(f.name).suffix.lower()
            
            # Count video files
            if ext in [
                ".mp4", ".avi", ".mkv", ".mov", ".wmv", 
                ".flv", ".webm", ".mpeg", ".m4v",
            ]:
                vids += 1
                
            # Count image files
            elif ext in [
                ".jpg", ".jpeg", ".png", ".gif", 
                ".bmp", ".tiff", ".webp"
            ]:
                imgs += 1
                
            # Count PowerPoint files
            elif ext in [".ppt", ".pptx"]:
                ppts += 1
                
            # Count PDF files
            elif ext == ".pdf":
                pdfs += 1
        
        # Validate file count constraints
        if vids > 1 or ppts > 1 or pdfs > 1:
            return False, "Only one video or one PPT or one PDF allowed"
            
        if imgs > 10:
            return False, "Maximum 10 images allowed"
            
        # Validate file type mixing constraints
        if vids > 0 and (imgs > 0 or ppts > 0 or pdfs > 0):
            return False, "Cannot mix videos with other media types"
            
        if (ppts > 0 or pdfs > 0) and imgs > 0:
            return False, "Cannot mix documents with images"
        
        return True, ""
    
    def process_files(self, files: List[Any]) -> List[Dict[str, Any]]:
        """
        Process uploaded media files and convert them to model-compatible format.
        
        This method handles different file types:
        - Videos: Added directly as video objects
        - Images: Added directly as image objects  
        - PowerPoint: Converted to images via PDF
        - PDF: Converted to individual page images
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            List of content objects for model input
        """
        out = []
        
        # Process each uploaded file
        for f in files or []:
            ext = Path(f.name).suffix.lower()
            
            # Handle video files
            if ext in [
                ".mp4", ".avi", ".mkv", ".mov", ".wmv", 
                ".flv", ".webm", ".mpeg", ".m4v",
            ]:
                out.append({"type": "video", "url": f.name})
                
            # Handle image files
            elif ext in [
                ".jpg", ".jpeg", ".png", ".gif", 
                ".bmp", ".tiff", ".webp"
            ]:
                out.append({"type": "image", "url": f.name})
                
            # Handle PowerPoint files (convert to images)
            elif ext in [".ppt", ".pptx"]:
                for p in self._ppt_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
                    
            # Handle PDF files (convert to images)
            elif ext == ".pdf":
                for p in self._pdf_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
                    
        return out
    
    @spaces.GPU(duration=240)  # Hugging Face Spaces GPU allocation decorator
    def stream_generate(
        self, 
        conversation_history: List[Dict[str, Any]], 
        system_prompt: str
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from the model.
        
        This method:
        1. Prepares conversation messages for model input
        2. Applies chat template and tokenizes input
        3. Configures generation parameters
        4. Runs generation in a separate thread for streaming
        5. Yields formatted text fragments as they're generated
        
        Args:
            conversation_history: List of conversation messages
            system_prompt: System prompt for the conversation
            
        Yields:
            Formatted text fragments during generation
        """
        self.stop_generation_flag = False
        
        # Build message list from conversation history
        msgs = self._build_messages(conversation_history, system_prompt)
        
        # Apply chat template and tokenize input
        inputs = self.processor.apply_chat_template(
            msgs,
            tokenize=True,              # Convert to tokens
            add_generation_prompt=True, # Add prompt for generation
            return_dict=True,           # Return as dictionary
            return_tensors="pt",        # Return PyTorch tensors
        ).to(self.model.device)         # Move to model device
        
        # Remove token_type_ids if present (not needed for generation)
        inputs.pop("token_type_ids", None)
        
        # Set up streaming tokenizer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True,           # Don't include input prompt in output
            skip_special_tokens=False   # Keep special tokens for processing
        )
        
        # Configure generation parameters
        gen_args = dict(
            inputs,
            max_new_tokens=8192,        # Maximum tokens to generate
            repetition_penalty=1.1,     # Penalty for repetition
            do_sample=True,             # Enable sampling
            top_k=2,                    # Top-k sampling
            temperature=None,           # No temperature scaling
            top_p=1e-5,                # Very low top-p for focused generation
            streamer=streamer,          # Text streaming handler
        )

        # Start generation in separate thread for non-blocking streaming
        generation_thread = threading.Thread(target=self.model.generate, kwargs=gen_args)
        generation_thread.start()

        # Stream generated tokens and format output
        buf = ""
        for tok in streamer:
            if self.stop_generation_flag:  # Check for early stopping
                break
            buf += tok
            yield self._stream_fragment(buf)  # Format and yield fragment

        generation_thread.join()  # Wait for generation to complete
    
    def stop_generation(self) -> None:
        """
        Stop the current generation process.
        """
        self.stop_generation_flag = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the GLM-4V model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": "GLM-4V",
            "description": "GLM-4.1V-9B-Thinking and GLM-4.5V Vision-Language Models",
            "supported_formats": ["text", "image", "video", "pdf", "ppt"],
            "max_images": 10,
            "max_videos": 1,
            "model_path": self.model_path
        }
    
    # Private helper methods
    
    def _strip_html(self, t: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            t: Input text potentially containing HTML tags
            
        Returns:
            Clean text with HTML tags removed
        """
        return re.sub(r"<[^>]+>", "", t).strip()

    def _wrap_text(self, t: str) -> List[Dict[str, str]]:
        """
        Wrap plain text in the expected message format.
        
        Args:
            t: Plain text string
            
        Returns:
            List containing a single text message object
        """
        return [{"type": "text", "text": t}]

    def _pdf_to_imgs(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to individual image files.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to generated image files
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing but not available")
            
        doc = fitz.open(pdf_path)  # Open PDF document
        imgs = []
        
        # Convert each page to an image
        for i in range(doc.page_count):
            # Render page as pixmap with 180 DPI for good quality
            pix = doc.load_page(i).get_pixmap(dpi=180)
            
            # Generate unique filename for each page
            img_p = os.path.join(
                tempfile.gettempdir(), 
                f"{Path(pdf_path).stem}_{i}.png"
            )
            
            # Save pixmap as PNG image
            pix.save(img_p)
            imgs.append(img_p)
            
        doc.close()  # Clean up PDF document
        return imgs

    def _ppt_to_imgs(self, ppt_path: str) -> List[str]:
        """
        Convert PowerPoint presentation to image files.
        
        This method uses LibreOffice to convert PPT/PPTX to PDF first,
        then converts the PDF pages to individual images.
        
        Args:
            ppt_path: Path to the PowerPoint file
            
        Returns:
            List of paths to generated image files
        """
        # Create temporary directory for conversion
        tmp = tempfile.mkdtemp()
        
        # Use LibreOffice headless mode to convert PPT to PDF
        subprocess.run(
            [
                "libreoffice",      # LibreOffice command
                "--headless",       # Run without GUI
                "--convert-to",     # Conversion mode
                "pdf",              # Target format
                "--outdir",         # Output directory
                tmp,                # Temporary directory
                ppt_path,           # Source PPT file
            ],
            check=True,  # Raise exception on failure
        )
        
        # Generate PDF path and convert to images
        pdf_path = os.path.join(tmp, Path(ppt_path).stem + ".pdf")
        return self._pdf_to_imgs(pdf_path)

    def _stream_fragment(self, buf: str) -> str:
        """
        Process streaming text fragments and format thinking/answer sections.
        
        The GLM-4V-Thinking model outputs structured responses with:
        - <think>...</think>: Internal reasoning process
        - <answer>...</answer>: Final response content
        
        This method formats these sections with appropriate HTML styling.
        
        Args:
            buf: Current accumulated text buffer from streaming
            
        Returns:
            Formatted HTML content for display
        """
        think_html = ""
        
        # Process thinking section
        if "<think>" in buf:
            if "</think>" in buf:
                # Complete thinking section - extract and format
                seg = re.search(r"<think>(.*?)</think>", buf, re.DOTALL)
                if seg:
                    think_html = (
                        "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                        "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                        + seg.group(1).strip().replace("\n", "<br>")
                        + "</div></details>"
                    )
            else:
                # Incomplete thinking section - show partial content
                part = buf.split("<think>", 1)[1]
                think_html = (
                    "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                    "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                    + part.replace("\n", "<br>")
                    + "</div></details>"
                )

        answer_html = ""
        
        # Process answer section
        if "<answer>" in buf:
            if "</answer>" in buf:
                # Complete answer section - extract content
                seg = re.search(r"<answer>(.*?)</answer>", buf, re.DOTALL)
                if seg:
                    answer_html = seg.group(1).strip()
            else:
                # Incomplete answer section - show partial content
                answer_html = buf.split("<answer>", 1)[1]

        # Return formatted content or clean HTML if no special sections
        if not think_html and not answer_html:
            return self._strip_html(buf)
        return think_html + answer_html

    def _build_messages(self, raw_hist: List[Dict[str, Any]], sys_prompt: str) -> List[Dict[str, Any]]:
        """
        Build message list for model input from conversation history.
        
        This method:
        1. Adds system prompt if provided
        2. Processes conversation history
        3. Cleans assistant messages by removing thinking sections and HTML
        4. Formats messages in the expected model input format
        
        Args:
            raw_hist: Raw conversation history
            sys_prompt: System prompt text
            
        Returns:
            Formatted message list for model input
        """
        msgs = []

        # Add system prompt if provided
        if sys_prompt.strip():
            msgs.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt.strip()}],
                }
            )

        # Process conversation history
        for h in raw_hist:
            if h["role"] == "user":
                # User messages are added as-is (may contain multimodal content)
                msgs.append({"role": "user", "content": h["content"]})
            else:
                # Clean assistant messages for model input
                raw = h["content"]
                # Remove thinking sections and HTML formatting
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
                raw = re.sub(r"<details.*?</details>", "", raw, flags=re.DOTALL)
                clean = self._strip_html(raw).strip()
                msgs.append({"role": "assistant", "content": self._wrap_text(clean)})
                
        return msgs