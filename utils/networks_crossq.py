from typing import Any, Optional, Sequence
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from flax.linen.module import Module, compact, merge_param
from jax.nn import initializers

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
        variable_axes={'params': 0, 'intermediates': 0, 'batch_stats': 0},
        split_rngs={'params': True, 'batch_stats': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class BatchRenorm(nn.Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper."""
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    dtype: Any = None
    param_dtype: Any = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Any = initializers.zeros
    scale_init: Any = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable('batch_stats', 'r_max', lambda s: s, 3)
        d_max = self.variable('batch_stats', 'd_max', lambda s: s, 5)
        steps = self.variable('batch_stats', 'steps', lambda s: s, 0)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
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
        use_batch_norm: Whether to apply batch normalization.
        batch_norm_momentum: Batch norm momentum.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99

    @nn.compact
    def __call__(self, x, training: bool = True):
        for i, size in enumerate(self.hidden_dims):
            if self.use_batch_norm and i == 0:
                x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
                
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.use_batch_norm:
                    x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
                elif self.layer_norm:
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
        use_batch_norm: Whether to apply batch normalization.
        batch_norm_momentum: Batch norm momentum.
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
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims, 
            activate_final=True, 
            layer_norm=self.layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum
        )
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
        training: bool = True,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
            training: Whether in training mode.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs, training=training)

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


class QNetwork(nn.Module):
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    hidden_dims: Sequence[int] = (512, 512)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = True):
        x = jnp.concatenate([observations, actions], -1)
        
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
        
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            x = nn.relu(x)
            if self.use_batch_norm:
                x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
            
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class DualQNetwork(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    encoder: nn.Module = None
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    crossq_style: bool = False

    def setup(self):
        self.q1_net = QNetwork(
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            hidden_dims=self.hidden_dims
        )
        self.q2_net = QNetwork(
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            hidden_dims=self.hidden_dims
        )

    def __call__(self, observations, actions=None, training: bool = True):
        if self.encoder is not None:
            observations = self.encoder(observations)
            
        if actions is None:
            actions = jnp.zeros_like(observations[..., :1])
        q1 = self.q1_net(observations, actions, training=training)
        q2 = self.q2_net(observations, actions, training=training)

        return jnp.stack([q1, q2], axis=0)


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2  
    encoder: nn.Module = None
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    crossq_style: bool = False

    def setup(self):
        if self.num_ensembles != 2:
            raise ValueError("This implementation only supports two Q networks.")
            
        self.dual_q = DualQNetwork(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            encoder=self.encoder,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            crossq_style=self.crossq_style
        )

    def __call__(self, observations, actions=None, training: bool = True):
        return self.dual_q(observations, actions, training=training)


# Fix 3: Helper function for consistent network calls with batch norm
def call_network_with_batch_norm(network, network_name, batch_stats, observations, 
                                  actions=None, training=True, params=None, **kwargs):
    """Helper function to call networks consistently with batch norm support."""
    if params is None:
        # Use network's own params
        if hasattr(network, 'batch_stats') and batch_stats is not None:
            # Use apply method with batch stats
            variables = {'params': network.params[f'modules_{network_name}'], 
                        'batch_stats': batch_stats[f'modules_{network_name}']}
            if actions is not None:
                result = network.apply_fn(variables, observations, actions, training=training, 
                                        mutable=['batch_stats'], **kwargs)
            else:
                result = network.apply_fn(variables, observations, training=training,
                                        mutable=['batch_stats'], **kwargs)
            if isinstance(result, tuple):
                output, new_batch_stats = result
                return output, new_batch_stats.get('batch_stats', {})
            else:
                return result, {}
        else:
            # No batch stats, use direct call
            if actions is not None:
                return network.select(network_name)(observations, actions, training=training, **kwargs), {}
            else:
                return network.select(network_name)(observations, training=training, **kwargs), {}
    else:
        # Use provided params
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats[f'modules_{network_name}']}
            if actions is not None:
                result = network.apply_fn(variables, observations, actions, training=training,
                                        mutable=['batch_stats'], **kwargs)
            else:
                result = network.apply_fn(variables, observations, training=training,
                                        mutable=['batch_stats'], **kwargs)
            if isinstance(result, tuple):
                output, new_batch_stats = result
                return output, new_batch_stats.get('batch_stats', {})
            else:
                return result, {}
        else:
            # No batch stats, use direct call with params
            if actions is not None:
                return network.select(network_name)(observations, actions, params=params, training=training, **kwargs), {}
            else:
                return network.select(network_name)(observations, params=params, training=training, **kwargs), {}


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
        use_batch_norm: Whether to apply batch normalization.
        batch_norm_momentum: Batch norm momentum.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum
        )
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False, training: bool = True):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
            training: Whether in training mode.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs, training=training)

        return v
    

class ActorVectorFieldGRU(nn.Module):
    """Actor vector field network for flow matching with GRU-style updates.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
        use_batch_norm: Whether to apply batch normalization.
        batch_norm_momentum: Batch norm momentum.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    hidden_dim_gru: int = 256
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum
        )
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)
        
        # GRU-style gate network with batch norm support
        self.gate_net = nn.Sequential([
            nn.Dense(self.hidden_dim_gru),
            nn.swish,
            nn.Dense(self.action_dim, 
                    kernel_init=zeros,
                    bias_init=constant(5.0)),
        ])
        
        if self.use_batch_norm:
            self.bn_input = BatchRenorm(momentum=self.batch_norm_momentum)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False, training: bool = True):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
            training: Whether in training mode.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        # Apply input batch norm if enabled
        if self.use_batch_norm:
            inputs = self.bn_input(inputs, use_running_average=not training)

        v = self.mlp(inputs, training=training)
        z = nn.sigmoid(self.gate_net(inputs))
        vector_field = z * (v - actions)
        return vector_field