import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from utils import logger as logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PpoConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    action_size: int = 9
    device: str = 'cpu'
    seed: int = 0
    a_max: float = 0.4
    w_max: float = 0.5235987755982988
    # learning_rate: float = 0.002
    # gradient_clip: float = 1.0
    # gamma = 0.99
    # eps_clip = 0.2


def encoder(input_dimension: int, output_dimension: int) -> nn.Sequential:
    """Create encoder model"""
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.LayerNorm(output_dimension),
        nn.ReLU()
    )

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim,action_max):
        super(Actor, self).__init__()
        self.mean_layer=nn.Sequential(
            nn.Linear(state_dim,PpoConfig.hidden_dim),
            nn.ReLU(),
            nn.Linear(PpoConfig.hidden_dim,PpoConfig.hidden_dim),
            nn.ReLU(),
            nn.Linear(PpoConfig.hidden_dim,action_dim)
        )
        self.log_std_layer = nn.Sequential(
            nn.Linear(state_dim,PpoConfig.hidden_dim),
            nn.ReLU(),
            nn.Linear(PpoConfig.hidden_dim,action_dim)
        )
        self.action_max = action_max
    def forward(self, state):
        mean = self.mean_layer(state)*self.action_max
        log_std = self.log_std_layer(state)
        log_std = torch.clamp(log_std,-1,1)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean,std)
    
class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim,PpoConfig.hidden_dim),
            nn.ReLU(),
            nn.Linear(PpoConfig.hidden_dim,PpoConfig.hidden_dim),
            nn.ReLU(),
            nn.Linear(PpoConfig.hidden_dim,1)
        )
    def forward(self, state):
        net = self.net(state)
        return net



class TargetSelectionModule(nn.Module):
    """Attention module specifically for target selection of evaders"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.self_transform = nn.Linear(2 * hidden_dim, hidden_dim)

        # Linear layer for transforming query
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

        # Linear layers for transforming key and value
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)

        # Layer Norm for feature fusion
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Scaling factor
        self.scale = math.sqrt(hidden_dim)

    def forward(self, self_feature: torch.Tensor, evader_features: torch.Tensor,
                evader_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_feature: Self features [batch_size, 2 * hidden_dim]
            evader_features: Evader features [batch_size, num_evaders, hidden_dim]
            evader_mask: Evader mask [batch_size, num_evaders]

        Returns:
            enhanced_feature: Enhanced features [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, num_evaders]
        """
        batch_size = self_feature.shape[0]  # [B, 2H]
        hidden_dim = self.hidden_dim
        num_evaders = evader_features.shape[1]

        # Add shape checking
        assert self_feature.shape == (batch_size, 2 * hidden_dim), f"Unexpected self_feature shape: {self_feature.shape}"
        assert evader_features.shape == (
            batch_size, num_evaders, hidden_dim), f"Unexpected evader_features shape: {evader_features.shape}"
        assert evader_mask.shape == (batch_size, num_evaders), f"Unexpected evader_mask shape: {evader_mask.shape}"

        self_feature = self.self_transform(self_feature)  # [B, H] # Transform self features

        # Transform query (self features)
        query = self.query_transform(self_feature).unsqueeze(1)  # [B, 1, H]

        # Transform key and value (evader features)
        keys = self.key_transform(evader_features)  # [B, N, H]
        values = self.value_transform(evader_features)  # [B, N, H]

        # Calculate attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # [B, 1, N]

        # Validate scores shape
        assert scores.shape == (batch_size, 1, num_evaders), f"Unexpected scores shape: {scores.shape}"

        # Apply mask
        if evader_mask is not None:
            scores = scores.masked_fill((~evader_mask.bool()).unsqueeze(1), float('-inf'))

        # Get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, N]

        # Get weighted features
        weighted_features = torch.matmul(attention_weights, values)  # [B, 1, H]
        weighted_features = weighted_features.squeeze(1)  # [B, H]

        # Feature fusion and normalization
        enhanced_feature = self.layer_norm(self_feature + weighted_features)

        # Final shape checking
        assert enhanced_feature.shape == (batch_size, hidden_dim)
        assert attention_weights.squeeze(1).shape == (batch_size, num_evaders)

        return enhanced_feature, attention_weights.squeeze(1)
    

class PpoPolicy(nn.Module):
    def __init__(self, config: Optional[PpoConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = PpoConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.action_dim = 2
        self.action_dim = [config.a_max,config.w_max]
        self.device = config.device
        torch.manual_seed(config.seed)

        self.feature_dims = {
            'self': 4,  # [vx, vy, min_obs_dis, pursuing_signal]
            'pursuers': 7,  # [px, py, vx, vy, dist, angle, pursuing_signal]
            'evaders': 7,  # [px, py, vx, vy, dist, pos_angle, head_angle]
            'obstacles': 5  # [px, py, radius, dist, angle]
        }
        
        self.actor = Actor(self.hidden_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.hidden_dim).to(self.device)
        self.entity_encoders = nn.ModuleDict({
            name: encoder(dim, self.hidden_dim)
            for name, dim in self.feature_dims.items()
        })

        self.type_embedding = nn.Embedding(4, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.hidden_dim,
            nhead = config.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout = 0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False
        )

        self.target_selection = TargetSelectionModule(self.hidden_dim)

        self.to(self.device)

        # For storing the last attention weights
        self._last_target_weights = None
        self._last_transformer_weights = None

    
    def _validate_input(self, obs: Dict[str, torch.Tensor]) -> None:
        """Validate the format and dimensions of input data"""
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")

        required_keys = {'self', 'types', 'masks'}
        if not all(key in obs for key in required_keys):
            raise ValueError(f"obs must contain keys: {required_keys}")

        if obs['self'].ndim != 2:
            raise ValueError("self features must be 2-dimensional [batch_size, feature_dim]")

        if obs['self'].shape[1] != self.feature_dims['self']:
            raise ValueError(f"self features must have dimension {self.feature_dims['self']}")

    def encoded_entities(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = obs['self'].shape[0]
        encoded_features = []

        for entity_type, encoder in self.entity_encoders.items():
            features = obs[entity_type]
            if entity_type == 'self':
                encoded = encoder(features).unsqueeze(1)
            else:
                encoded = encoder(features)
            encoded_features.append(encoded)

        entity_embed = torch.cat(encoded_features, dim=1)
        type_embed = self.type_embedding(obs['types'].long())
        tokens = entity_embed+type_embed

        attention_mask = ~obs['masks'].bool()
        transformed = self.transformer_encoder(tokens, src_key_padding_mask=attention_mask)

        self_feature = transformed[:,0]
        global_feature = torch.max(transformed,dim=1).values

        enhanced_feature = torch.cat([self_feature, global_feature], dim=-1) 

        evader_indices = (obs['types']==2)
        num_evaders_per_batch = evader_indices.sum(dim=1)  # [B]
        max_evaders = num_evaders_per_batch.max().item()
        flat_mask = obs['masks'].masked_select(evader_indices)
        flat_features = transformed.masked_select(
            evader_indices.unsqueeze(-1).expand(-1, -1, transformed.size(-1))
        )

        evader_mask = flat_mask.reshape(B, max_evaders)  # [B, max_evaders]
        evader_features = flat_features.reshape(B, max_evaders, -1) 

        enhanced_features, attention_weights = self.target_selection(
            enhanced_feature,
            evader_features,
            evader_mask
        )

        self._last_target_weights = attention_weights
        return enhanced_features
    
    def forward(self, obs: Dict[str, torch.Tensor])->Tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(obs)

        features = self.encoded_entities(obs)
        # action_probs, values = self.policy(features)
        action_probs = self.actor(features)
        values = self.critic(features)
        return action_probs, values

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get the attention weights from the last forward pass"""
        return {
            'target_selection': self._last_target_weights,
            'transformer': self._last_transformer_weights
        }

    def count_parameters(self) -> int:
        """Count the number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: str) -> None:
        """
        Save the model, using version numbers to avoid overwriting existing files

        Args:
            directory: Save directory
        """
        try:
            os.makedirs(directory, exist_ok=True)

            # Find existing version numbers in the directory
            existing_versions = []
            for filename in os.listdir(directory):
                if filename.startswith("network_params_v") and filename.endswith(".pth"):
                    try:
                        version = int(filename[len("network_params_v"):-4])
                        existing_versions.append(version)
                    except ValueError:
                        continue

            # Determine the new version number
            new_version = max(existing_versions, default=0) + 1

            # Save model parameters with the new version number
            params_path = os.path.join(directory, f"network_params_v{new_version}.pth")
            torch.save(self.state_dict(), params_path)

            # Save the corresponding version configuration file
            config_path = os.path.join(directory, f"config_v{new_version}.json")
            with open(config_path, 'w') as f:
                json.dump(vars(self.config), f)

            logger.info(f"Model saved as version {new_version} in {directory}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(cls,
             directory: str,
             device: str = 'cpu',
             version: Optional[int] = None,
             **kwargs) -> 'PpoPolicy':
        """
        Load the model

        Args:
            directory: Model directory
            version: Specific version number to load, if None then load the latest version
            device: Device
            **kwargs: Other parameters

        Returns:
            TERLPolicy: Loaded model

        Raises:
            FileNotFoundError: When the specified version's model file does not exist
            ValueError: When there are no valid model files in the directory
        """
        try:
            # Get all version numbers
            versions = []
            for filename in os.listdir(directory):
                if filename.startswith("network_params_v") and filename.endswith(".pth"):
                    try:
                        ver = int(filename[len("network_params_v"):-4])
                        versions.append(ver)
                    except ValueError:
                        continue

            if not versions:
                raise ValueError(f"No valid model files found in {directory}")

            # Determine the version to load
            if version is None:
                version = max(versions)  # Load the latest version
            elif version not in versions:
                raise FileNotFoundError(f"Version {version} not found in {directory}")

            params_path = os.path.join(directory, f"network_params_v{version}.pth")
            config_path = os.path.join(directory, f"config_v{version}.json")

            # Load configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = PpoConfig(**config_dict)
            else:
                logger.warning(f"Config file for version {version} not found, using default configuration")
                config = PpoConfig()

            # Update device and other parameters
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Create model and load parameters
            model = cls(config)
            model.load_state_dict(torch.load(params_path, map_location=device))
            model.to(device)

            logger.info(f"Successfully loaded model version {version} from {directory}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise










