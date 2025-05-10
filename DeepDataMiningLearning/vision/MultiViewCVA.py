#Integrate multi-camera input (e.g., front, side, rear views) into a pretrained VLA model while minimizing changes to its original architecture and avoiding full model retraining.
#augment the input token stream of the VLA model with extra visual tokens from multiple camera views. 
#Using a learned embedding layer to encode multi-view inputs before they enter the model.

# Multi-View Camera Vision Alignment (CVA)
# Purpose: Integrate multi-camera input (e.g., front, side, rear views) into a pretrained VLA model 
# while minimizing changes to its original architecture and avoiding full model retraining.
# Strategy: Augment the input token stream of the VLA model with extra visual tokens from multiple camera views
# using a learned embedding layer to encode multi-view inputs before they enter the model.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import time
from nuscenes.nuscenes import NuScenes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiCameraNuScenesDataset(Dataset):
    """
    Dataset class for loading multi-camera images from NuScenes dataset.
    Each sample contains images from 6 cameras around the vehicle.
    """
    def __init__(self, nusc, transform=None, visualize_samples=False, vis_dir='output/nuscenes_vis'):
        """
        Initialize the dataset with NuScenes instance and optional transforms.
        
        Args:
            nusc: NuScenes dataset instance
            transform: Optional transforms to apply to images
            visualize_samples: Whether to visualize and save sample images
            vis_dir: Directory to save visualizations
        """
        self.nusc = nusc  # NuScenes dataset instance
        self.samples = [s for s in nusc.sample]  # Get all samples from NuScenes
        self.camera_names = ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),  # Resize to standard input size
            transforms.ToTensor()  # Convert to tensor and normalize to [0,1]
        ])
        self.visualize_samples = visualize_samples
        self.vis_dir = vis_dir
        
        # Create visualization directory if needed
        if self.visualize_samples and not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
            
        logger.info(f"Initialized dataset with {len(self.samples)} samples and {len(self.camera_names)} cameras")

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (images, token) where:
                - images: Tensor of shape (6, 3, H, W) containing all camera views
                - token: Sample token for identification
        """
        sample = self.samples[idx]  # Get sample at index
        images = []
        raw_images = []  # Store raw images for visualization
        
        # Load images from all cameras
        for cam in self.camera_names:
            # Get camera token and image path
            cam_token = sample['data'][cam]
            img_path = self.nusc.get('sample_data', cam_token)['filename']
            img_full_path = os.path.join(self.nusc.dataroot, img_path)
            
            # Read and convert image
            img = cv2.imread(img_full_path)
            if img is None:
                raise ValueError(f"Failed to load image at {img_full_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            raw_images.append(img.copy())  # Save raw image for visualization
            
            # Apply transforms
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        # Visualize if enabled
        if self.visualize_samples and idx < 5:  # Visualize first 5 samples
            self._visualize_sample(raw_images, sample['token'], idx)
            
        # Stack all camera images into a single tensor (6, 3, H, W)
        return torch.stack(images), sample['token']
    
    def _visualize_sample(self, images, token, idx):
        """
        Visualize a sample with all camera views.
        
        Args:
            images: List of raw camera images
            token: Sample token
            idx: Sample index
        """
        # Create figure with subplots for each camera
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Plot each camera view
        for i, (img, cam_name) in enumerate(zip(images, self.camera_names)):
            ax = fig.add_subplot(gs[i//3, i%3])
            ax.imshow(img)
            ax.set_title(cam_name.replace('CAM_', ''))
            ax.axis('off')
        
        plt.suptitle(f"Sample {idx}: {token}", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(self.vis_dir, f"sample_{idx}_{token}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved visualization to {save_path}")


class MultiViewEmbeddingEncoder(nn.Module):
    """
    Neural network model that encodes multiple camera views into a unified embedding space.
    Uses a shared backbone encoder followed by view-specific projections and fusion.
    Supports various backbone architectures including ResNet, Vision Transformers, and Hugging Face models.
    """
    def __init__(self, 
                 output_dim=768, 
                 backbone_type='resnet18', 
                 pretrained=True, 
                 num_views=6, 
                 fusion_dim=768, 
                 fusion_layers=1,
                 use_efficient_attention=True):
        """
        Initialize the multi-view encoder.
        
        Args:
            output_dim: Dimension of output embeddings (should match VLA model dimension)
            backbone_type: Type of backbone to use ('resnet18', 'resnet50', 'vit_b_16', 'vit_l_16', 
                          or a Hugging Face model name like 'google/vit-base-patch16-224')
            pretrained: Whether to use pretrained weights for the encoder
            num_views: Number of camera views to process
            fusion_dim: Hidden dimension for the transformer fusion module
            fusion_layers: Number of transformer layers for fusion
            use_efficient_attention: Whether to use memory-efficient attention implementation
        """
        super().__init__()
        
        # Initialize backbone based on type
        self.backbone_type = backbone_type
        self.backbone_dim = 0  # Will be set based on backbone
        
        if backbone_type.startswith('resnet'):
            # ResNet from torchvision
            if backbone_type == 'resnet18':
                self.encoder = models.resnet18(pretrained=pretrained)
                self.backbone_dim = 512
            elif backbone_type == 'resnet34':
                self.encoder = models.resnet34(pretrained=pretrained)
                self.backbone_dim = 512
            elif backbone_type == 'resnet50':
                self.encoder = models.resnet50(pretrained=pretrained)
                self.backbone_dim = 2048
            elif backbone_type == 'resnet101':
                self.encoder = models.resnet101(pretrained=pretrained)
                self.backbone_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet type: {backbone_type}")
                
            # Remove classification head
            self.encoder.fc = nn.Identity()
            
        elif backbone_type.startswith('vit'):
            # Vision Transformer from torchvision
            if backbone_type == 'vit_b_16':
                self.encoder = models.vit_b_16(pretrained=pretrained)
                self.backbone_dim = 768
            elif backbone_type == 'vit_b_32':
                self.encoder = models.vit_b_32(pretrained=pretrained)
                self.backbone_dim = 768
            elif backbone_type == 'vit_l_16':
                self.encoder = models.vit_l_16(pretrained=pretrained)
                self.backbone_dim = 1024
            elif backbone_type == 'vit_l_32':
                self.encoder = models.vit_l_32(pretrained=pretrained)
                self.backbone_dim = 1024
            else:
                raise ValueError(f"Unsupported ViT type: {backbone_type}")
                
            # Replace head with identity
            self.encoder.heads = nn.Identity()
            
        else:
            # Try to load from Hugging Face
            try:
                from transformers import AutoModel, AutoFeatureExtractor
                
                self.encoder = AutoModel.from_pretrained(backbone_type)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(backbone_type)
                
                # Determine output dimension from model config
                if hasattr(self.encoder.config, 'hidden_size'):
                    self.backbone_dim = self.encoder.config.hidden_size
                else:
                    # Default for many vision models
                    self.backbone_dim = 768
                    
                logger.info(f"Loaded Hugging Face model {backbone_type} with dimension {self.backbone_dim}")
                
                # Flag to indicate this is a Hugging Face model
                self.is_huggingface = True
            except Exception as e:
                logger.error(f"Failed to load Hugging Face model: {e}")
                raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Freeze encoder weights to avoid modifying the pretrained backbone
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info(f"Initialized frozen {backbone_type} backbone with output dimension {self.backbone_dim}")

        # View-specific projection layers (backbone_dim -> output_dim)
        self.view_projectors = nn.ModuleList([
            nn.Linear(self.backbone_dim, output_dim) for _ in range(num_views)
        ])
        
        # Learnable view position embeddings
        self.view_embeddings = nn.Parameter(torch.randn(num_views, output_dim))
        
        # Fusion module: transformer encoder to integrate information across views
        # Use memory-efficient attention if requested
        if use_efficient_attention:
            # Use PyTorch's memory-efficient attention if available (PyTorch 2.0+)
            if hasattr(nn.TransformerEncoderLayer, 'norm_first'):
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=output_dim,  # Embedding dimension
                    nhead=8,  # Number of attention heads
                    dim_feedforward=fusion_dim,  # Hidden dimension in feedforward network
                    batch_first=False,  # Input is (seq_len, batch, dim)
                    norm_first=True,  # Pre-normalization for better stability
                    activation='gelu',  # GELU activation (used in modern transformers)
                )
            else:
                # Fallback for older PyTorch versions
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=output_dim,
                    nhead=8,
                    dim_feedforward=fusion_dim,
                    batch_first=False,
                    activation='gelu'
                )
        else:
            # Try to use Hugging Face's efficient implementation
            try:
                from transformers.models.bert.modeling_bert import BertLayer
                
                class HFTransformerEncoderLayer(nn.Module):
                    def __init__(self, d_model, nhead, dim_feedforward):
                        super().__init__()
                        self.bert_layer = BertLayer(
                            config=type('obj', (object,), {
                                'hidden_size': d_model,
                                'num_attention_heads': nhead,
                                'intermediate_size': dim_feedforward,
                                'hidden_dropout_prob': 0.1,
                                'attention_probs_dropout_prob': 0.1,
                                'layer_norm_eps': 1e-12,
                                'add_cross_attention': False
                            })
                        )
                        
                    def forward(self, x, attention_mask=None):
                        # Convert from (seq_len, batch, dim) to (batch, seq_len, dim)
                        x = x.permute(1, 0, 2)
                        outputs = self.bert_layer(x, attention_mask=attention_mask)
                        # Convert back to (seq_len, batch, dim)
                        return outputs[0].permute(1, 0, 2)
                
                encoder_layer = HFTransformerEncoderLayer(
                    d_model=output_dim,
                    nhead=8,
                    dim_feedforward=fusion_dim
                )
                
                # Custom TransformerEncoder to work with our layer
                import copy
                class CustomTransformerEncoder(nn.Module):
                    def __init__(self, encoder_layer, num_layers):
                        super().__init__()
                        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
                        
                    def forward(self, x, mask=None):
                        for layer in self.layers:
                            x = layer(x, attention_mask=mask)
                        return x
                
                self.fusion = CustomTransformerEncoder(encoder_layer, num_layers=fusion_layers)
                logger.info("Using Hugging Face's efficient transformer implementation")
                
            except ImportError:
                # Fallback to standard PyTorch implementation
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=output_dim,
                    nhead=8,
                    dim_feedforward=fusion_dim,
                    batch_first=False
                )
                self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
                logger.info("Using PyTorch's standard transformer implementation")
        
        # VLA model integration (to be implemented)
        self.vla_model = None
        logger.info(f"Initialized MultiViewEmbeddingEncoder with {num_views} views and {output_dim}-dim embeddings")

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, V, C, H, W) where:
               B = batch size, V = number of views, C = channels, H = height, W = width
               
        Returns:
            Tensor of shape (B, V, output_dim) containing fused embeddings for all views
        """
        B, V, C, H, W = x.shape
        
        # Process each view separately through the backbone
        features = []
        
        for v in range(V):
            view_input = x[:, v]  # (B, C, H, W)
            
            # Process with appropriate backbone
            if hasattr(self, 'is_huggingface') and self.is_huggingface:
                # For Hugging Face models, we need to use their preprocessing
                # Convert to list of PIL images for feature extractor
                pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 
                             for img in view_input]
                
                # Process with feature extractor
                inputs = self.feature_extractor(pil_images, return_tensors="pt").to(view_input.device)
                
                # Get embeddings from model
                with torch.no_grad():
                    outputs = self.encoder(**inputs)
                
                # Use the pooled output or last hidden state
                if hasattr(outputs, 'pooler_output'):
                    view_features = outputs.pooler_output
                else:
                    # Use CLS token or mean of last hidden state
                    view_features = outputs.last_hidden_state[:, 0]
            else:
                # For torchvision models
                with torch.no_grad():
                    view_features = self.encoder(view_input)
            
            features.append(view_features)
        
        # Stack features for all views
        features = torch.stack(features, dim=1)  # (B, V, backbone_dim)
        
        # Project each view separately and add view-specific embedding
        projected = []
        for v in range(V):
            # Apply projection and add view embedding
            proj = self.view_projectors[v](features[:, v]) + self.view_embeddings[v]
            projected.append(proj)
            
        # Stack view projections
        tokens = torch.stack(projected, dim=1)  # (B, V, output_dim)
        
        # Fuse tokens using transformer
        # Transformer expects (seq_len, batch, dim) format
        fused = self.fusion(tokens.permute(1, 0, 2))  # (V, B, output_dim)
        
        # Return to (batch, seq_len, dim) format
        fused = fused.permute(1, 0, 2)  # (B, V, output_dim)
        
        return fused  # (B, V, output_dim)


def contrastive_alignment_loss(view_embeddings, temperature=0.07):
    """
    Compute contrastive alignment loss between different camera views.
    This encourages embeddings from the same scene but different views to be similar.
    
    Args:
        view_embeddings: Tensor of shape (B, V, D) containing embeddings for each view
        temperature: Temperature parameter for scaling similarity scores
        
    Returns:
        Scalar loss value
    """
    B, V, D = view_embeddings.shape
    loss = 0.0
    
    # Compare each pair of views
    for i in range(V):
        for j in range(V):
            if i == j:
                continue  # Skip self-comparison
                
            # Normalize embeddings to unit length
            zi = F.normalize(view_embeddings[:, i], dim=-1)  # (B, D)
            zj = F.normalize(view_embeddings[:, j], dim=-1)  # (B, D)
            
            # Compute similarity matrix
            logits = torch.matmul(zi, zj.T) / temperature  # (B, B)
            
            # Labels are the indices themselves (diagonal is positive pairs)
            labels = torch.arange(B).to(logits.device)
            
            # Cross-entropy loss (each row should match with corresponding row in other view)
            loss += F.cross_entropy(logits, labels)
    
    # Average over all view pairs
    return loss / (V * (V - 1))


def evaluate_model(model, loader, num_batches=5, visualize=False, vis_dir='output/similarity_vis'):
    """
    Evaluate the model by computing mean embedding similarity across views.
    
    Args:
        model: MultiViewEmbeddingEncoder model
        loader: DataLoader for evaluation
        num_batches: Number of batches to evaluate
        visualize: Whether to visualize similarity matrices
        vis_dir: Directory to save visualizations
        
    Returns:
        Mean similarity score
    """
    model.eval()
    total_sim = 0.0
    count = 0
    
    # Create visualization directory if needed
    if visualize and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    with torch.no_grad():
        for batch_idx, (imgs, tokens) in enumerate(loader):
            if batch_idx >= num_batches:
                break
                
            # Move to GPU if available
            imgs = imgs.to(next(model.parameters()).device)
            
            # Get embeddings
            embeddings = model(imgs)  # (B, V, D)
            
            # Compute similarity matrix between views
            # einsum computes dot product between all pairs of view embeddings
            similarities = torch.einsum('bvd,bwd->bvw', embeddings, embeddings)  # (B, V, V)
            
            # Visualize similarity matrix
            if visualize:
                visualize_similarity_matrix(
                    similarities[0].cpu().numpy(),  # Take first batch item
                    os.path.join(vis_dir, f"similarity_batch_{batch_idx}.png")
                )
            
            # Compute mean similarity
            mean_sim = similarities.mean().item()
            total_sim += mean_sim
            count += 1
            
    return total_sim / count if count > 0 else 0.0


def visualize_similarity_matrix(similarity_matrix, save_path):
    """
    Visualize a similarity matrix between camera views.
    
    Args:
        similarity_matrix: 2D numpy array of shape (V, V)
        save_path: Path to save the visualization
    """
    camera_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'FRONT_LEFT', 'FRONT_RIGHT']
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    
    # Add labels
    plt.xticks(np.arange(len(camera_names)), camera_names, rotation=45)
    plt.yticks(np.arange(len(camera_names)), camera_names)
    
    # Add colorbar and title
    plt.colorbar(label='Similarity')
    plt.title('Cross-View Embedding Similarity')
    
    # Add text annotations
    for i in range(len(camera_names)):
        for j in range(len(camera_names)):
            plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                     ha="center", va="center", color="w" if similarity_matrix[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved similarity matrix visualization to {save_path}")


def visualize_training_progress(losses, similarities, save_path='output/training_progress.png'):
    """
    Visualize training progress with loss and similarity curves.
    
    Args:
        losses: List of loss values per epoch
        similarities: List of similarity values per epoch
        save_path: Path to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss
    epochs = range(1, len(losses) + 1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, losses, 'o-', color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # Create second y-axis for similarity
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Similarity', color='tab:blue')
    ax2.plot(epochs, similarities, 's-', color='tab:blue', label='Similarity')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Add title and legend
    plt.title('Training Progress')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved training progress visualization to {save_path}")


def visualize_embeddings_tsne(embeddings, save_path='output/embeddings_tsne.png'):
    """
    Visualize embeddings using t-SNE dimensionality reduction.
    
    Args:
        embeddings: Tensor of shape (B, V, D) containing embeddings
        save_path: Path to save the visualization
    """
    from sklearn.manifold import TSNE
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy and reshape
    embeddings_np = embeddings.cpu().numpy()
    B, V, D = embeddings_np.shape
    embeddings_flat = embeddings_np.reshape(-1, D)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_flat)
    
    # Reshape back to separate batches and views
    embeddings_2d = embeddings_2d.reshape(B, V, 2)
    
    # Plot
    plt.figure(figsize=(12, 10))
    camera_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'FRONT_LEFT', 'FRONT_RIGHT']
    colors = plt.cm.rainbow(np.linspace(0, 1, V))
    
    for v in range(V):
        plt.scatter(
            embeddings_2d[:, v, 0], 
            embeddings_2d[:, v, 1],
            color=colors[v],
            label=camera_names[v],
            alpha=0.7
        )
    
    # Connect points from the same batch with lines
    for b in range(B):
        for v1 in range(V):
            for v2 in range(v1+1, V):
                plt.plot(
                    [embeddings_2d[b, v1, 0], embeddings_2d[b, v2, 0]],
                    [embeddings_2d[b, v1, 1], embeddings_2d[b, v2, 1]],
                    'k-', alpha=0.1
                )
    
    plt.title('t-SNE Visualization of Multi-View Embeddings')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved t-SNE visualization to {save_path}")


def train_and_evaluate(dataroot, output_dir='output/multi_view_cva', 
                       batch_size=4, epochs=10, lr=1e-4, 
                       visualize=True, save_model=True):
    """
    Train and evaluate the Multi-View Embedding Encoder model.
    
    Args:
        dataroot: Path to NuScenes dataset
        output_dir: Directory to save outputs
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        visualize: Whether to create visualizations
        save_model: Whether to save the trained model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logger
    logger.info(f"Starting training with dataroot={dataroot}, batch_size={batch_size}, epochs={epochs}")
    
    # Load NuScenes dataset
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
        logger.info(f"Loaded NuScenes dataset with {len(nusc.sample)} samples")
    except Exception as e:
        logger.error(f"Failed to load NuScenes dataset: {e}")
        print(f"Error: {e}")
        print("Please make sure the NuScenes dataset is available at the specified path.")
        return
    
    # Create dataset and data loader
    dataset = MultiCameraNuScenesDataset(
        nusc, 
        visualize_samples=visualize,
        vis_dir=os.path.join(output_dir, 'sample_vis')
    )
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = MultiViewEmbeddingEncoder(output_dim=768).to(device)
    
    # Only train new modules (embedding, projector, fusion)
    params_to_train = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_train.append(param)
    
    logger.info(f"Training {len(params_to_train)} parameter groups, {sum(p.numel() for p in params_to_train)} parameters")
    
    # Initialize optimizer
    optimizer = optim.Adam(params_to_train, lr=lr)
    
    # Training loop
    losses = []
    similarities = []
    best_similarity = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Training loop
        for imgs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            
            # Forward pass
            embeddings = model(imgs)
            
            # Compute loss
            loss = contrastive_alignment_loss(embeddings)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Compute average loss
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        losses.append(avg_loss)
        
        # Evaluate model
        mean_sim = evaluate_model(
            model, 
            loader, 
            visualize=visualize and (epoch == 0 or epoch == epochs-1),
            vis_dir=os.path.join(output_dir, f'similarity_epoch_{epoch+1}')
        )
        similarities.append(mean_sim)
        
        # Save best model
        if save_model and mean_sim > best_similarity:
            best_similarity = mean_sim
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"Saved best model with similarity {best_similarity:.4f}")
        
        # Log progress
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Mean Similarity={mean_sim:.4f}, Time={epoch_time:.2f}s")
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Mean Similarity={mean_sim:.4f}, Time={epoch_time:.2f}s")
    
    # Visualize training progress
    if visualize:
        visualize_training_progress(losses, similarities, os.path.join(output_dir, 'training_progress.png'))
        
        # Visualize embeddings with t-SNE
        with torch.no_grad():
            model.eval()
            for imgs, _ in loader:
                imgs = imgs.to(device)
                embeddings = model(imgs)
                visualize_embeddings_tsne(embeddings, os.path.join(output_dir, 'embeddings_tsne.png'))
                break  # Only visualize one batch
    
    # Save final model
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'similarities': similarities,
            'epochs': epochs,
        }, os.path.join(output_dir, 'final_model.pth'))
        logger.info(f"Saved final model to {os.path.join(output_dir, 'final_model.pth')}")
    
    return model, losses, similarities


def load_pretrained_vla_model(model_path_or_name="openvla/openvla-7b"):
    """
    Load a pretrained VLA (Vision-Language-Action) model from Hugging Face.
    
    Args:
        model_path_or_name: Path to the pretrained model or Hugging Face model name
                           Default is "openvla/openvla-7b"
        
    Returns:
        Tuple of (model, processor) for the loaded VLA model
    """
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    logger.info(f"Loading pretrained VLA model from {model_path_or_name}")
    
    try:
        # Check if model_path is a local path or a Hugging Face model name
        if os.path.exists(model_path_or_name):
            processor = AutoProcessor.from_pretrained(model_path_or_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_path_or_name,
                attn_implementation="flash_attention_2",  # Optional, requires flash_attn
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info(f"Loaded VLA model from local path: {model_path_or_name}")
        else:
            # Load from Hugging Face
            processor = AutoProcessor.from_pretrained(model_path_or_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_path_or_name,
                attn_implementation="flash_attention_2",  # Optional, requires flash_attn
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info(f"Loaded VLA model from Hugging Face: {model_path_or_name}")
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        return model, processor
    
    except Exception as e:
        logger.error(f"Failed to load VLA model: {e}")
        logger.info("If loading fails, make sure you have installed the required dependencies:")
        logger.info("pip install transformers timm tokenizers flash-attn")
        raise


def integrate_with_vla(multi_view_model, vla_model, vla_processor, sample_input, text_prompt=None):
    """
    Integrate multi-view embeddings with a VLA model.
    
    Args:
        multi_view_model: Trained MultiViewEmbeddingEncoder
        vla_model: Pretrained VLA model
        vla_processor: Processor for the VLA model
        sample_input: Sample input with multiple camera views
        text_prompt: Optional text prompt for multimodal models
        
    Returns:
        VLA model output (action prediction)
    """
    logger.info("Integrating multi-view embeddings with VLA model")
    
    # Get multi-view embeddings
    multi_view_embeddings = multi_view_model(sample_input)  # Shape: (B, V, output_dim)
    batch_size, num_views, embedding_dim = multi_view_embeddings.shape
    
    # Default prompt if none provided
    if text_prompt is None:
        text_prompt = "In: What action should the robot take to complete the task?\nOut:"
    
    # Process the first camera view with VLA processor
    # This is for getting the original VLA vision embeddings format
    first_view = sample_input[:, 0]  # Take the first view (B, C, H, W)
    
    # Process with VLA processor
    inputs = vla_processor(text=text_prompt, images=first_view, return_tensors="pt").to(multi_view_embeddings.device)
    
    # Method 1: Replace the vision embeddings with our multi-view embeddings
    # This requires understanding the VLA model architecture
    
    # For OpenVLA, we can use the predict_action method directly
    with torch.no_grad():
        # Get the original vision embeddings
        vision_outputs = vla_model.get_vision_outputs(inputs["pixel_values"])
        
        # Replace with our multi-view embeddings (average pooling across views)
        pooled_embeddings = multi_view_embeddings.mean(dim=1)  # (B, output_dim)
        
        # Format to match VLA's expected input
        # This may need adjustment based on the specific VLA model architecture
        if hasattr(vla_model, "predict_action"):
            # Use the model's built-in action prediction
            action = vla_model.predict_action(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                vision_hidden_states=pooled_embeddings,
                do_sample=False
            )
            return action
        else:
            # Generic approach - forward through the model
            outputs = vla_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                vision_hidden_states=pooled_embeddings
            )
            return outputs


def evaluate_vla_integration(multi_view_model, vla_model, vla_processor, test_loader, 
                            num_samples=5, text_prompts=None, output_dir='output/vla_evaluation'):
    """
    Evaluate the integration of multi-view embeddings with a VLA model on test data.
    
    Args:
        multi_view_model: Trained MultiViewEmbeddingEncoder
        vla_model: Pretrained VLA model
        vla_processor: Processor for the VLA model
        test_loader: DataLoader for test data
        num_samples: Number of samples to evaluate
        text_prompts: List of text prompts to test (for multimodal models)
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set models to evaluation mode
    multi_view_model.eval()
    vla_model.eval()
    
    # Default prompts if none provided
    if text_prompts is None:
        text_prompts = [
            "In: What action should the robot take to pick up the object?\nOut:",
            "In: What action should the robot take to navigate around the obstacle?\nOut:",
            "In: What action should the robot take to open the drawer?\nOut:"
        ]
    
    # Prepare for evaluation
    results = []
    device = next(multi_view_model.parameters()).device
    
    # Evaluate on test data
    with torch.no_grad():
        for i, (imgs, tokens) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Move to device
            imgs = imgs.to(device)
            
            # Process with different text prompts
            for prompt_idx, prompt in enumerate(text_prompts):
                logger.info(f"Testing sample {i+1}/{num_samples} with prompt: {prompt}")
                
                try:
                    # Integrate and get outputs (actions)
                    action = integrate_with_vla(
                        multi_view_model, 
                        vla_model, 
                        vla_processor,
                        imgs, 
                        text_prompt=prompt
                    )
                    
                    # Save results
                    result = {
                        'sample_idx': i,
                        'token': tokens[0],  # Assuming batch size 1 for simplicity
                        'prompt': prompt,
                        'action': action.cpu().numpy().tolist() if isinstance(action, torch.Tensor) else action
                    }
                    results.append(result)
                    
                    # Visualize results
                    visualize_vla_action_prediction(
                        imgs[0].cpu(),  # First item in batch
                        action,
                        prompt,
                        os.path.join(output_dir, f"sample_{i}_prompt_{prompt_idx}.png")
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing sample {i} with prompt '{prompt}': {e}")
                    continue
    
    # Save all results
    import json
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    return results


def visualize_vla_action_prediction(images, action, prompt, save_path):
    """
    Visualize the action prediction from the VLA model.
    
    Args:
        images: Input images tensor of shape (V, C, H, W)
        action: Predicted action from the VLA model
        prompt: Text prompt used
        save_path: Path to save the visualization
    """
    # Create figure
    num_views = images.shape[0]
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid for camera views and action visualization
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot input images
    for v in range(min(num_views, 6)):  # Show up to 6 views
        ax = fig.add_subplot(gs[v//3, v%3])
        # Convert tensor to numpy for visualization
        img = images[v].permute(1, 2, 0).numpy()
        # Normalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(f"View {v+1}")
        ax.axis('off')
    
    # Add title with prompt
    plt.suptitle(f"Prompt: {prompt}", fontsize=16)
    
    # Add action visualization
    if isinstance(action, torch.Tensor):
        action_np = action.cpu().numpy()
    elif isinstance(action, np.ndarray):
        action_np = action
    elif isinstance(action, list):
        action_np = np.array(action)
    else:
        # Handle other types or just display the text representation
        action_text = str(action)
        plt.figtext(0.5, 0.01, f"Action: {action_text}", ha="center", fontsize=12)
        action_np = None
    
    # If we have a numeric action array, visualize it
    if action_np is not None and action_np.size > 0:
        # Add a subplot for action visualization
        ax = fig.add_subplot(gs[1, :])
        
        # If action is a vector, plot as bar chart
        if action_np.ndim == 1 or (action_np.ndim == 2 and action_np.shape[0] == 1):
            action_flat = action_np.flatten()
            ax.bar(range(len(action_flat)), action_flat)
            ax.set_xlabel('Action Dimension')
            ax.set_ylabel('Value')
            ax.set_title('Predicted Action Vector')
        # If action is a sequence, plot as line
        elif action_np.ndim == 2:
            for i in range(action_np.shape[0]):
                ax.plot(action_np[i], label=f'Sequence {i+1}')
            ax.legend()
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.set_title('Predicted Action Sequence')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved action prediction visualization to {save_path}")


def test_vla_models(model_names=None, multi_view_model_path=None, test_data_path=None, 
                   output_dir='output/vla_tests', num_samples=3):
    """
    Test multiple VLA models with the multi-view encoder.
    
    Args:
        model_names: List of Hugging Face model names to test
        multi_view_model_path: Path to the trained multi-view model
        test_data_path: Path to test data
        output_dir: Directory to save test results
        num_samples: Number of samples to test per model
    """
    # Default model names if none provided
    if model_names is None:
        model_names = ["openvla/openvla-7b"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load multi-view model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if multi_view_model_path is None:
        logger.info("No multi-view model path provided, initializing a new model")
        multi_view_model = MultiViewEmbeddingEncoder().to(device)
    else:
        logger.info(f"Loading multi-view model from {multi_view_model_path}")
        multi_view_model = MultiViewEmbeddingEncoder().to(device)
        multi_view_model.load_state_dict(torch.load(multi_view_model_path, map_location=device))
    
    multi_view_model.eval()
    
    # Load test data
    if test_data_path is None:
        logger.error("No test data path provided")
        return {"error": "No test data path provided"}
    
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=test_data_path, verbose=True)
        dataset = MultiCameraNuScenesDataset(nusc)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        logger.info(f"Loaded test dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return {"error": f"Failed to load test data: {e}"}
    
    # Define test prompts
    test_prompts = [
        "In: What action should the robot take to pick up the object?\nOut:",
        "In: What action should the robot take to navigate around the obstacle?\nOut:",
        "In: What action should the robot take to open the drawer?\nOut:"
    ]
    
    # Test each model
    results = {}
    for model_name in model_names:
        logger.info(f"Testing model: {model_name}")
        model_output_dir = os.path.join(output_dir, model_name.replace('/', '_'))
        os.makedirs(model_output_dir, exist_ok=True)
        
        try:
            # Load VLA model
            vla_model, vla_processor = load_pretrained_vla_model(model_name)
            
            # Evaluate
            model_results = evaluate_vla_integration(
                multi_view_model,
                vla_model,
                vla_processor,
                loader,
                num_samples=num_samples,
                text_prompts=test_prompts,
                output_dir=model_output_dir
            )
            
            results[model_name] = {
                "status": "success",
                "num_samples_processed": len(model_results),
                "output_dir": model_output_dir
            }
            
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}")
            results[model_name] = {"status": "error", "error": str(e)}
    
    # Save overall results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    logger.info(f"Testing complete. Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test VLA models with multi-view inputs')
    parser.add_argument('--model', type=str, default="openvla/openvla-7b",
                        help='Hugging Face model name or path')
    parser.add_argument('--multi_view_model', type=str, default=None,
                        help='Path to trained multi-view model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to NuScenes dataset')
    parser.add_argument('--output_dir', type=str, default='output/vla_tests',
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to test')
    # New arguments for enhanced functionality
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'train', 'evaluate'],
                        help='Operation mode: test, train, or evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='Backbone architecture for MultiViewEmbeddingEncoder')
    parser.add_argument('--fusion_dim', type=int, default=768,
                        help='Dimension for transformer fusion module')
    parser.add_argument('--fusion_layers', type=int, default=1,
                        help='Number of transformer layers for fusion')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization during training and evaluation')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints during training')
    
    args = parser.parse_args()
    
    # Execute the appropriate function based on the mode
    if args.mode == 'test':
        # Test a single model
        test_vla_models(
            model_names=[args.model],
            multi_view_model_path=args.multi_view_model,
            test_data_path=args.data_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    elif args.mode == 'train':
        # Create and train the MultiViewEmbeddingEncoder
        train_and_evaluate(
            dataroot=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            visualize=args.visualize,
            save_model=args.save_model
        )
    elif args.mode == 'evaluate':
        # Load NuScenes dataset
        try:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version='v1.0-mini', dataroot=args.data_path, verbose=True)
            dataset = MultiCameraNuScenesDataset(nusc)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            
            # Load models
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load multi-view model
            multi_view_model = MultiViewEmbeddingEncoder(
                output_dim=768,
                backbone_type=args.backbone,
                fusion_dim=args.fusion_dim,
                fusion_layers=args.fusion_layers
            ).to(device)
            
            if args.multi_view_model:
                multi_view_model.load_state_dict(torch.load(args.multi_view_model, map_location=device))
                logger.info(f"Loaded multi-view model from {args.multi_view_model}")
            
            # Load VLA model
            vla_model, vla_processor = load_pretrained_vla_model(args.model)
            
            # Evaluate
            results = evaluate_vla_integration(
                multi_view_model,
                vla_model,
                vla_processor,
                loader,
                num_samples=args.num_samples,
                output_dir=args.output_dir
            )
            
            logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    else:
        logger.error(f"Unknown mode: {args.mode}")