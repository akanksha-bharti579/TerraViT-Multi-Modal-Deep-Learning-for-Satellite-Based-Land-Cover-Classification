"""
Self-supervised contrastive learning framework for satellite imagery.
Implements SimCLR-style pre-training for multi-modal feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used for contrastive self-supervised learning.
    
    Args:
        temp: Temperature parameter for scaling similarities
        reduction_mode: How to aggregate batch losses
    """
    
    def __init__(self, temp=0.07, reduction_mode='mean'):
        super().__init__()
        self.temp = temp
        self.reduction_mode = reduction_mode
    
    def forward(self, embeddings):
        """
        Compute contrastive loss for a batch of embeddings.
        
        Args:
            embeddings: Tensor of shape (2*batch_size, embedding_dim)
                       where first half and second half are augmented pairs
                       Example: embeddings[0] and embeddings[batch_size] are a pair
        
        Returns:
            torch.Tensor: Scalar loss value (mean or sum based on reduction_mode)
        """
        # Validate input shape
        if embeddings.shape[0] % 2 != 0:
            raise ValueError(
                f"Embeddings batch size must be even, got {embeddings.shape[0]}"
            )
        
        batch_sz = embeddings.shape[0] // 2
        
        # L2 normalize embeddings to unit sphere for cosine similarity
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute pairwise cosine similarity matrix
        # Shape: (2*batch_size, 2*batch_size)
        # Each element (i,j) is cosine similarity between embeddings[i] and embeddings[j]
        similarity = torch.matmul(embeddings, embeddings.T) / self.temp
        
        # Create masks for positive and negative pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i) for i in [0, batch_size)
        pos_mask = self._create_positive_mask(batch_sz).to(embeddings.device)
        # Negative pairs: all pairs except self and positive pairs
        neg_mask = self._create_negative_mask(batch_sz).to(embeddings.device)
        
        # Extract positive pair similarities (one per sample)
        # Each sample has one positive pair (its augmented version)
        pos_sim = torch.sum(similarity * pos_mask, dim=1)
        
        # Compute log-sum-exp of negative similarities for InfoNCE loss
        # Apply exponential to all similarities, mask out non-negatives
        neg_sim = torch.exp(similarity) * neg_mask
        neg_sum = torch.sum(neg_sim, dim=1)  # Sum over all negatives
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # Simplified: -pos_sim + log(sum(exp(neg)) + epsilon)
        # Small epsilon prevents log(0)
        loss_per_sample = -pos_sim + torch.log(neg_sum + 1e-8)
        
        # Aggregate loss across batch
        if self.reduction_mode == 'mean':
            return loss_per_sample.mean()
        else:
            return loss_per_sample.sum()
    
    def _create_positive_mask(self, batch_sz):
        """
        Generate mask identifying positive pairs in similarity matrix.
        
        Positive pairs are augmented versions of the same sample:
        - Sample i and sample (i + batch_sz) are positive pairs
        - Mask is symmetric: mask[i, j] = mask[j, i]
        
        Args:
            batch_sz: Number of unique samples (before augmentation)
            
        Returns:
            torch.Tensor: Binary mask of shape (2*batch_sz, 2*batch_sz)
                         where 1 indicates positive pairs, 0 otherwise
        """
        mask = torch.zeros(2 * batch_sz, 2 * batch_sz, dtype=torch.float32)
        
        # Mark positive pairs: each sample i is paired with its augmented version
        for i in range(batch_sz):
            # Pair (i, i+batch_sz) - original with its augmentation
            mask[i, batch_sz + i] = 1.0
            # Pair (i+batch_sz, i) - symmetric relationship
            mask[batch_sz + i, i] = 1.0
        
        return mask
    
    def _create_negative_mask(self, batch_sz):
        """
        Generate mask identifying negative pairs in similarity matrix.
        
        Negative pairs are all pairs except:
        - Self-similarity (diagonal elements)
        - Positive pairs (augmented versions of same sample)
        
        Args:
            batch_sz: Number of unique samples (before augmentation)
            
        Returns:
            torch.Tensor: Binary mask of shape (2*batch_sz, 2*batch_sz)
                         where 1 indicates negative pairs, 0 otherwise
        """
        # Start with all pairs as negatives
        mask = torch.ones(2 * batch_sz, 2 * batch_sz, dtype=torch.float32)
        
        # Remove self-similarity (diagonal elements)
        # A sample should not be compared with itself
        mask = mask - torch.eye(2 * batch_sz, dtype=torch.float32)
        
        # Remove positive pairs (augmented versions of same sample)
        for i in range(batch_sz):
            # Remove (i, i+batch_sz) - this is a positive pair
            mask[i, batch_sz + i] = 0.0
            # Remove (i+batch_sz, i) - symmetric positive pair
            mask[batch_sz + i, i] = 0.0
        
        return mask


class MultiModalProjector(nn.Module):
    """
    Projection head for mapping backbone features to embedding space.
    Typically used on top of CNN/Transformer encoders.
    
    Args:
        input_features: Dimensionality of input features from backbone
        projection_size: Size of output embedding space
        hidden_size: Size of intermediate layer
    """
    
    def __init__(self, input_features, projection_size=128, hidden_size=2048):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    
    def forward(self, features):
        """Project features to embedding space"""
        return self.projection(features)


class DualStreamContrastiveModel(nn.Module):
    """
    Two-tower model for learning aligned representations across modalities.
    Each tower processes one modality and projects to shared embedding space.
    
    Args:
        backbone1: Neural network for first modality (e.g., SAR)
        backbone2: Neural network for second modality (e.g., optical)
        projection_dim: Dimensionality of final embeddings
    """
    
    def __init__(self, backbone1, backbone2, projection_dim=128):
        super().__init__()
        
        self.encoder_modality1 = backbone1
        self.encoder_modality2 = backbone2
        
        # Assume backbones output 2048-d features (ResNet50 default)
        self.projector1 = MultiModalProjector(2048, projection_dim)
        self.projector2 = MultiModalProjector(2048, projection_dim)
    
    def forward(self, inputs_mod1, inputs_mod2):
        """
        Process inputs from both modalities through their respective towers.
        
        Args:
            inputs_mod1: Batch from first modality
            inputs_mod2: Batch from second modality
            
        Returns:
            Tuple of (embeddings_mod1, embeddings_mod2)
        """
        # Extract features
        features1 = self.encoder_modality1(inputs_mod1)
        features2 = self.encoder_modality2(inputs_mod2)
        
        # Project to embedding space
        embeddings1 = self.projector1(features1)
        embeddings2 = self.projector2(features2)
        
        return embeddings1, embeddings2
    
    def get_modality1_features(self, inputs):
        """Extract features from first modality only"""
        return self.encoder_modality1(inputs)
    
    def get_modality2_features(self, inputs):
        """Extract features from second modality only"""
        return self.encoder_modality2(inputs)


class ContrastiveTrainer:
    """
    Training coordinator for self-supervised contrastive learning.
    Handles forward/backward passes and optimization.
    
    Args:
        model: Contrastive model (DualStreamContrastiveModel)
        optimizer: PyTorch optimizer
        lr_scheduler: Learning rate scheduler
        config: Training configuration dictionary
    """
    
    def __init__(self, model, optimizer, lr_scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        
        self.criterion = InfoNCELoss(temp=config.get('temperature', 0.07))
        self.device = config.get('device', 'cuda')
        self.model.to(self.device)
    
    def train_epoch(self, dataloader):
        """
        Execute one training epoch.
        
        Args:
            dataloader: PyTorch DataLoader yielding batches
                Each batch should be a dict with keys 'modality1' and 'modality2'
            
        Returns:
            dict: Dictionary with training metrics:
                - 'loss': Average loss over the epoch
                - 'lr': Current learning rate
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            # Extract modality data and move to appropriate device (CPU/GPU)
            mod1_data = batch_data['modality1'].to(self.device)
            mod2_data = batch_data['modality2'].to(self.device)
            
            # Forward pass: extract embeddings from both modalities
            # emb1: embeddings from modality 1 (batch, projection_dim)
            # emb2: embeddings from modality 2 (batch, projection_dim)
            emb1, emb2 = self.model(mod1_data, mod2_data)
            
            # Concatenate embeddings for contrastive loss
            # Creates pairs: [emb1[0], emb1[1], ..., emb2[0], emb2[1], ...]
            # Where emb1[i] and emb2[i] are positive pairs
            all_embeddings = torch.cat([emb1, emb2], dim=0)  # (2*batch, projection_dim)
            
            # Compute InfoNCE contrastive loss
            loss = self.criterion(all_embeddings)
            
            # Backward pass: compute gradients
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients via backpropagation
            self.optimizer.step()  # Update model parameters
            
            # Update learning rate according to scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Track metrics for logging
            total_loss += loss.item()
            num_batches += 1
        
        # Compute average loss over all batches
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'lr': self.optimizer.param_groups[0]['lr']}
    
    def save_checkpoint(self, filepath, epoch, metrics):
        """
        Save model checkpoint to disk.
        
        Args:
            filepath: Path where checkpoint will be saved
            epoch: Current training epoch number
            metrics: Dictionary of training metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),  # Model parameters
            'optimizer_state': self.optimizer.state_dict(),  # Optimizer state (momentum, etc.)
            'metrics': metrics,  # Training metrics (loss, accuracy, etc.)
            'config': self.config  # Training configuration for reproducibility
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint from disk.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            tuple: (epoch, metrics) where epoch is the saved epoch number
                   and metrics is the saved metrics dictionary
                   
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            KeyError: If checkpoint is missing required keys
        """
        # Load checkpoint to same device as model
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Return saved epoch and metrics
        return checkpoint['epoch'], checkpoint['metrics']

