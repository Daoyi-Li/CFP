from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen.initializers import zeros, constant

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
    
class ActorVectorFieldGRU(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    hidden_dim_gru: int = 256
    denoising_steps: int = 4

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)
        self.gate_net = nn.Sequential([
            nn.Dense(self.hidden_dim_gru),
            nn.swish,  # Mish approximation
            nn.Dense(self.action_dim, 
            kernel_init=zeros,
            bias_init=constant(5.0)),
        ])


    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)
        # v = 1*nn.tanh(v)
        z = nn.sigmoid(self.gate_net(inputs))
        z = z * self.denoising_steps
        vector_field = z * (v - actions)
        # vector_field = z * (v - 0.01*actions)
        return vector_field

class ActorVectorFieldGRUFinetune(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    hidden_dim_gru: int = 256
    denoising_steps: int = 4
    noise: float = 0.05

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)
        self.gate_net = nn.Sequential([
            nn.Dense(self.hidden_dim_gru),
            nn.swish,  # Mish approximation
            nn.Dense(self.action_dim, 
            kernel_init=zeros,
            bias_init=constant(5.0)),
        ])


    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)
        # s_t = (t_safe * u_t - xt) / (1 - t_safe)
        drift_coef = v + (0.5*self.noise**2) * (times * v - actions) / (1 - times)
        # v = 1*nn.tanh(v)
        z = nn.sigmoid(self.gate_net(inputs))
        z = z * self.denoising_steps
        vector_field = z * (drift_coef)
        return vector_field

class ActorVectorFieldEmb(nn.Module):
    """Actor vector field network with Transformer-style embeddings.
    
    Uses the same embedding approach as FlowMatchingTransformerActor:
    - Observations are encoded through a 2-layer MLP
    - Actions are projected through a Dense layer
    - Times are encoded through a 3-layer MLP
    - Action and time embeddings are added, then concatenated with observation embedding
    
    Attributes:
        hidden_dims: Hidden layer dimensions for final MLP.
        action_dim: Action dimension.
        d_model: Embedding dimension (like Transformer's d_model).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    d_model: int = 128  # Embedding dimension, matching Transformer default
    
    def setup(self) -> None:
        # Observation encoder (2-layer MLP to d_model) - same as Transformer
        self.obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Action projection (single Dense layer to d_model) - same as Transformer
        self.action_proj = nn.Dense(self.d_model, name='action_proj')
        
        # Time embedding (3-layer MLP to d_model) - same as Transformer
        self.time_embedding = nn.Sequential([
            nn.Dense(self.d_model // 4),
            nn.silu,
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Final MLP to predict velocity
        # Input dimension: d_model (obs) + d_model (action + time combined) = 2 * d_model
        self.output_mlp = MLP(
            (*self.hidden_dims, self.action_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm
        )
    
    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times.
        
        Args:
            observations: Observations.
            actions: Actions.
            times: Times (required for proper embedding combination).
            is_encoded: Whether the observations are already encoded.
        """
        # Encode observations if needed
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        
        # Create embeddings following Transformer approach
        obs_emb = self.obs_encoder(observations)
        action_emb = self.action_proj(actions)
        
        if times is not None:
            time_emb = self.time_embedding(times)
            # Combine action and time embeddings (additive like in Transformer)
            action_time_emb = action_emb + time_emb
        else:
            # If no time provided, just use action embedding
            action_time_emb = action_emb
        
        # Concatenate observation embedding with action-time embedding
        # This creates the final input representation
        combined_emb = jnp.concatenate([obs_emb, action_time_emb], axis=-1)
        
        # Pass through final MLP to get velocity prediction
        v = self.output_mlp(combined_emb)
        
        return v


# class ActorVectorFieldEmb(nn.Module):
#     """Actor vector field network with Transformer-style embeddings.
    
#     Uses the same embedding approach as FlowMatchingTransformerActor:
#     - Observations are encoded through a 2-layer MLP
#     - Actions are projected through a Dense layer
#     - Times are encoded through a 3-layer MLP
#     - Action and time embeddings are added, then concatenated with observation embedding
    
#     Attributes:
#         hidden_dims: Hidden layer dimensions for final MLP.
#         action_dim: Action dimension.
#         d_model: Embedding dimension (like Transformer's d_model).
#         layer_norm: Whether to apply layer normalization.
#         encoder: Optional encoder module to encode the inputs.
#     """

#     hidden_dims: Sequence[int]
#     action_dim: int
#     layer_norm: bool = False
#     encoder: nn.Module = None
#     use_fourier_features: bool = False
#     fourier_feature_dim: int = 64
#     d_model: int = 128  # Embedding dimension, matching Transformer default
    
#     def setup(self) -> None:
#         # Observation encoder (2-layer MLP to d_model) - same as Transformer
#         self.obs_encoder = nn.Sequential([
#             nn.Dense(self.d_model // 2),
#             nn.silu,
#             nn.Dense(self.d_model)
#         ])
        
#         # Action projection (single Dense layer to d_model) - same as Transformer
#         self.action_proj = nn.Dense(self.d_model, name='action_proj')
        
#         # Time embedding (3-layer MLP to d_model) - same as Transformer
#         self.time_embedding = nn.Sequential([
#             nn.Dense(self.d_model // 4),
#             nn.silu,
#             nn.Dense(self.d_model // 2),
#             nn.silu,
#             nn.Dense(self.d_model)
#         ])
        
#         # Final MLP to predict velocity
#         # Input dimension: d_model (obs) + d_model (action + time combined) = 2 * d_model
#         self.output_mlp = MLP(
#             (*self.hidden_dims, self.action_dim), 
#             activate_final=False, 
#             layer_norm=self.layer_norm
#         )
    
#     @nn.compact
#     def __call__(self, observations, actions, times=None, is_encoded=False):
#         """Return the vectors at the given states, actions, and times.
        
#         Args:
#             observations: Observations.
#             actions: Actions.
#             times: Times (required for proper embedding combination).
#             is_encoded: Whether the observations are already encoded.
#         """
#         # Encode observations if needed
#         if not is_encoded and self.encoder is not None:
#             observations = self.encoder(observations)
        
#         # Create embeddings following Transformer approach
#         obs_emb = self.obs_encoder(observations)
#         action_emb = self.action_proj(actions)
        
#         if times is not None:
#             time_emb = self.time_embedding(times)
#             # Combine action and time embeddings (additive like in Transformer)
#             action_time_emb = action_emb + time_emb
#         else:
#             # If no time provided, just use action embedding
#             action_time_emb = action_emb
        
#         # Concatenate observation embedding with action-time embedding
#         # This creates the final input representation
#         combined_emb = jnp.concatenate([obs_emb, action_time_emb], axis=-1)
        
#         # Pass through final MLP to get velocity prediction
#         v = self.output_mlp(combined_emb)
        
#         return v

class ActorVectorFieldGRUEmb(nn.Module):
    """Actor vector field network with GRU-style gating and Transformer-style embeddings.
    
    Combines the embedding approach from ActorVectorFieldEmb with the gating mechanism
    from ActorVectorFieldGRUFinetune.
    
    Attributes:
        hidden_dims: Hidden layer dimensions for final MLP.
        action_dim: Action dimension.
        d_model: Embedding dimension (like Transformer's d_model).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
        hidden_dim_gru: Hidden dimension for gate network.
        denoising_steps: Number of denoising steps.
        noise: Noise level for drift coefficient calculation.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    d_model: int = 128  # Embedding dimension
    hidden_dim_gru: int = 256
    denoising_steps: int = 4
    noise: float = 0.05
    
    def setup(self) -> None:
        # Observation encoder (2-layer MLP to d_model) - same as ActorVectorFieldEmb
        self.obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Action projection (single Dense layer to d_model) - same as ActorVectorFieldEmb
        self.action_proj = nn.Dense(self.d_model, name='action_proj')
        
        # Time embedding (3-layer MLP to d_model) - same as ActorVectorFieldEmb
        self.time_embedding = nn.Sequential([
            nn.Dense(self.d_model // 4),
            nn.silu,
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Final MLP to predict velocity
        # Input dimension: d_model (obs) + d_model (action + time combined) = 2 * d_model
        self.output_mlp = MLP(
            (*self.hidden_dims, self.action_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm
        )
        
        # Gate network - takes the combined embedding as input
        # Input dimension: 2 * d_model (same as MLP input)
        self.gate_net = nn.Sequential([
            nn.Dense(self.hidden_dim_gru),
            nn.swish,  # Mish approximation
            nn.Dense(self.action_dim, 
                     kernel_init=zeros,  
                     bias_init=constant(5.0)),
        ])
    
    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times.
        
        Args:
            observations: Observations.
            actions: Actions.
            times: Times (required for proper embedding combination and drift calculation).
            is_encoded: Whether the observations are already encoded.
        """
        # Encode observations if needed
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        
        # Create embeddings following Transformer approach (same as ActorVectorFieldEmb)
        obs_emb = self.obs_encoder(observations)
        action_emb = self.action_proj(actions)
        
        if times is not None:
            time_emb = self.time_embedding(times)
            # Combine action and time embeddings (additive like in Transformer)
            action_time_emb = action_emb + time_emb
        else:
            # If no time provided, just use action embedding
            action_time_emb = action_emb
        
        # Concatenate observation embedding with action-time embedding
        # This creates the final input representation
        combined_emb = jnp.concatenate([obs_emb, action_time_emb], axis=-1)
        
        # Pass through MLP to get base velocity prediction
        v = self.output_mlp(combined_emb)
        
        # Calculate drift coefficient (same as GRUFinetune)
        if times is not None:
            # s_t = (t_safe * u_t - xt) / (1 - t_safe)
            drift_coef = v + (0.5 * self.noise**2) * (times * v - actions) / (1 - times)
        else:
            drift_coef = v
        
        # Calculate gating coefficient (same as GRUFinetune)
        z = nn.sigmoid(self.gate_net(combined_emb))
        z = z * self.denoising_steps
        
        # Apply gating to drift coefficient
        vector_field = z * drift_coef
        
        return vector_field