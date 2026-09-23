import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

# 导入Actor保存/加载函数
from utils.save_actor_utils import save_actor_params, save_actor_params_lightweight, load_actor_params
from utils.save_critic_utils import save_critic_params, load_critic_into_agent
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'cube-quadruple', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'square-mh-low_dim', 'Environment (dataset) name.')

flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_bool('use_q_in_offline', True, "define whether to update through q in offline learning")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

# Actor保存相关flags
flags.DEFINE_bool('save_actor_on_eval', True, "save actor parameters during evaluation")
flags.DEFINE_string('actor_save_format', 'flax', "actor save format: 'flax' or 'pickle' or 'lightweight'")

# Actor加载相关flags
flags.DEFINE_string('load_actor_checkpoint', None, 'Path to pretrained actor checkpoint to load')
flags.DEFINE_bool('skip_offline_training', False, 'Skip offline training and load pretrained actor')
flags.DEFINE_integer('critic_warmup_steps', 10000, 'Number of steps to warmup critic before online training')
flags.DEFINE_bool('cross_architecture_load', False, 'Load from ActorVectorFieldEmb to Transformer architecture')

flags.DEFINE_integer('more_critic', 1, "define the more critic update")

# OGBench多数据集加载相关flags
flags.DEFINE_integer('num_datasets_for_buffer', 10, 'Number of OGBench datasets to randomly load for replay buffer initialization. -1 means use only the first dataset (default behavior), 0 means load all datasets, >0 means randomly sample that many datasets.')


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        if self.wandb_logger.run is not None:
            self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)
        else:
            pass


def save_actor_if_enabled(agent, save_dir, step, phase=""):
    """保存Actor参数（如果启用）"""
    if FLAGS.save_actor_on_eval:
        try:
            if FLAGS.actor_save_format == 'lightweight':
                save_path = save_actor_params_lightweight(agent, save_dir, step)
            else:
                save_path = save_actor_params(agent, save_dir, step, FLAGS.actor_save_format)
            
            print(f"[{phase}] Actor saved at step {step}: {save_path}")
        except Exception as e:
            print(f"[{phase}] Failed to save actor at step {step}: {e}")


def recursive_update_params(target_params, source_params, prefix="", updated_paths=None):
    """
    递归地将source_params中的叶子节点参数更新到target_params中
    （用于同架构加载）
    """
    if updated_paths is None:
        updated_paths = []
    
    if hasattr(target_params, 'unfreeze'):
        target_params = target_params.unfreeze()
    else:
        target_params = dict(target_params)
    
    for key, value in source_params.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, (dict, type(target_params))):
            if key in target_params:
                target_params[key], updated_paths = recursive_update_params(
                    target_params[key], 
                    value, 
                    current_path, 
                    updated_paths
                )
            else:
                print(f"  ⚠ Warning: Key '{current_path}' not found in target params")
        else:
            if key in target_params:
                target_params[key] = value
                updated_paths.append(current_path)
                print(f"  ✓ Updated leaf: {current_path}")
            else:
                print(f"  ⚠ Warning: Leaf '{current_path}' not found in target params")
    
    return target_params, updated_paths


def recursive_map_with_details(target_params, source_params, source_prefix, target_prefix, mapping_records):
    """
    递归映射参数并记录详细的映射关系
    
    Args:
        target_params: 目标参数字典
        source_params: 源参数字典
        source_prefix: 源路径前缀
        target_prefix: 目标路径前缀
        mapping_records: 映射记录列表
    
    Returns:
        updated_params: 更新后的目标参数
        mapping_records: 更新后的映射记录
    """
    if hasattr(target_params, 'unfreeze'):
        target_params = target_params.unfreeze()
    else:
        target_params = dict(target_params)
    
    for key, value in source_params.items():
        source_path = f"{source_prefix}.{key}" if source_prefix else key
        target_path = f"{target_prefix}.{key}" if target_prefix else key
        
        if isinstance(value, dict):
            if key in target_params:
                # 递归处理嵌套字典
                target_params[key], mapping_records = recursive_map_with_details(
                    target_params[key],
                    value,
                    source_path,
                    target_path,
                    mapping_records
                )
            else:
                print(f"    ⚠ Warning: Target key '{target_path}' not found")
        else:
            # 叶子节点 - 实际的参数（权重、偏置等）
            if key in target_params:
                # 获取参数形状信息
                source_shape = value.shape if hasattr(value, 'shape') else 'N/A'
                target_shape = target_params[key].shape if hasattr(target_params[key], 'shape') else 'N/A'
                
                # 检查形状是否匹配
                if source_shape == target_shape:
                    target_params[key] = value
                    mapping_records.append({
                        'source': source_path,
                        'target': target_path,
                        'shape': source_shape,
                        'status': '✓'
                    })
                else:
                    mapping_records.append({
                        'source': source_path,
                        'target': target_path,
                        'shape': f"Source: {source_shape}, Target: {target_shape}",
                        'status': '✗ SHAPE MISMATCH'
                    })
                    print(f"    ✗ Shape mismatch: {source_path} {source_shape} -> {target_path} {target_shape}")
            else:
                print(f"    ⚠ Warning: Target leaf '{target_path}' not found")
    
    return target_params, mapping_records


def map_embedding_params(source_emb_params, target_emb_params, source_name, target_name, mapping_records):
    """
    映射embedding参数并记录详细映射关系
    
    Args:
        source_emb_params: 源参数
        target_emb_params: 目标参数
        source_name: 源组件名称（如 'actor_onestep_flow.obs_encoder'）
        target_name: 目标组件名称（如 'actor_transformer.obs_encoder'）
        mapping_records: 映射记录列表
    
    Returns:
        updated_params: 更新后的目标参数
        mapping_records: 更新后的映射记录
    """
    print(f"\n  {'='*70}")
    print(f"  Mapping: {source_name} → {target_name}")
    print(f"  {'='*70}")
    
    target_emb_params, mapping_records = recursive_map_with_details(
        target_emb_params,
        source_emb_params,
        source_name,
        target_name,
        mapping_records
    )
    
    return target_emb_params, mapping_records


def print_mapping_summary(mapping_records):
    """打印映射摘要表格"""
    print(f"\n{'='*100}")
    print("PARAMETER MAPPING SUMMARY")
    print(f"{'='*100}")
    print(f"{'Status':<8} {'Source (ActorVectorFieldEmb)':<50} {'Target (Transformer)':<50}")
    print(f"{'-'*100}")
    
    for record in mapping_records:
        status = record['status']
        source = record['source']
        target = record['target']
        
        # 缩短路径以便显示
        if len(source) > 48:
            source = '...' + source[-45:]
        if len(target) > 48:
            target = '...' + target[-45:]
        
        print(f"{status:<8} {source:<50} {target:<50}")
        
        # 如果有形状不匹配，显示详细信息
        if '✗' in status:
            print(f"         Shape: {record['shape']}")
    
    print(f"{'-'*100}")
    
    # 统计
    success_count = sum(1 for r in mapping_records if r['status'] == '✓')
    fail_count = sum(1 for r in mapping_records if '✗' in r['status'])
    
    print(f"Total mapped: {success_count} parameters")
    if fail_count > 0:
        print(f"Failed: {fail_count} parameters (shape mismatch)")
    print(f"{'='*100}\n")


def analyze_transformer_structure(target_params):
    """
    分析Transformer网络结构，识别自动命名的Sequential组件
    
    由于Transformer使用@nn.compact，Sequential会被自动命名为Dense_0, Dense_1等
    我们需要根据结构推断哪些是obs_encoder, time_embedding, velocity_head
    
    Returns:
        dict: 包含识别出的组件映射 {'obs_encoder': key, 'time_embedding': key, 'velocity_head': key}
    """
    component_map = {}
    
    # 收集所有Dense和MLP组件
    dense_keys = [k for k in target_params.keys() if k.startswith('Dense_')]
    mlp_keys = [k for k in target_params.keys() if k.startswith('MLP_')]
    
    print(f"\n  Analyzing Transformer structure:")
    print(f"  - Found Dense layers: {dense_keys}")
    print(f"  - Found MLP layers: {mlp_keys}")
    print(f"  - Found other keys: {[k for k in target_params.keys() if not k.startswith('Dense_') and not k.startswith('MLP_') and not k.startswith('layer_')]}")
    
    # obs_encoder: 通常是前2个Dense层 (d_model//2, d_model)
    # time_embedding: 通常是接下来的3个Dense层 (d_model//4, d_model//2, d_model)
    # velocity_head: 通常是MLP_0或者最后的Dense层组
    
    # 按照Dense_数字排序
    dense_keys_sorted = sorted(dense_keys, key=lambda x: int(x.split('_')[1]))
    
    if len(dense_keys_sorted) >= 5:
        # obs_encoder: Dense_0, Dense_1
        component_map['obs_encoder'] = dense_keys_sorted[:2]
        # time_embedding: Dense_2, Dense_3, Dense_4
        component_map['time_embedding'] = dense_keys_sorted[2:5]
    elif len(dense_keys_sorted) >= 2:
        # 只有obs_encoder
        component_map['obs_encoder'] = dense_keys_sorted[:2]
    
    # velocity_head: MLP_0
    if mlp_keys:
        component_map['velocity_head'] = mlp_keys[0]
    
    print(f"\n  Identified components:")
    for comp_name, comp_keys in component_map.items():
        print(f"  - {comp_name}: {comp_keys}")
    
    return component_map


def map_sequential_params(source_params, target_params_dict, source_name, target_name, 
                          target_keys, mapping_records):
    """
    映射Sequential的参数到多个自动命名的Dense层
    
    Args:
        source_params: 源Sequential参数 (如 obs_encoder)
        target_params_dict: 目标参数字典
        source_name: 源名称
        target_name: 目标名称前缀
        target_keys: 目标Dense层的键列表 (如 ['Dense_0', 'Dense_1'])
        mapping_records: 映射记录
    """
    print(f"\n  {'='*70}")
    print(f"  Mapping: {source_name} → {target_name} ({target_keys})")
    print(f"  {'='*70}")
    
    # 如果source_params是Sequential，它的结构是: {'layers_0': {...}, 'layers_1': {...}}
    # 我们需要映射到target的 Dense_0, Dense_1等
    
    # 获取source中的layer keys并排序
    if isinstance(source_params, dict):
        source_layer_keys = sorted([k for k in source_params.keys() if k.startswith('layers_')])
    else:
        source_layer_keys = []
    
    if len(source_layer_keys) != len(target_keys):
        print(f"    ⚠ Warning: Layer count mismatch. Source: {len(source_layer_keys)}, Target: {len(target_keys)}")
    
    # 逐层映射
    for i, (source_key, target_key) in enumerate(zip(source_layer_keys, target_keys)):
        if source_key in source_params and target_key in target_params_dict:
            full_source_path = f"{source_name}.{source_key}"
            full_target_path = f"{target_name}.{target_key}"
            
            # 递归映射这一层的参数
            target_params_dict[target_key], mapping_records = recursive_map_with_details(
                target_params_dict[target_key],
                source_params[source_key],
                full_source_path,
                full_target_path,
                mapping_records
            )
    
    return target_params_dict, mapping_records


def load_cross_architecture_actor(checkpoint_path, agent_class, seed, 
                                   example_batch, config, save_format='flax'):
    """
    从ActorVectorFieldEmb加载参数到FlowMatchingTransformerActor
    
    映射关系：
    - actor_onestep_flow.obs_encoder -> actor_transformer.[Dense_0, Dense_1]
    - actor_onestep_flow.action_proj -> actor_transformer.action_proj
    - actor_onestep_flow.time_embedding -> actor_transformer.[Dense_2, Dense_3, Dense_4]
    - actor_onestep_flow.output_mlp -> actor_transformer.MLP_0 (velocity_head)
    """
    print(f"\n{'='*100}")
    print("CROSS-ARCHITECTURE PARAMETER LOADING")
    print(f"{'='*100}")
    print(f"Source Architecture: ActorVectorFieldEmb (actor_onestep_flow)")
    print(f"Target Architecture: FlowMatchingTransformerActor (actor_transformer)")
    print(f"Checkpoint Path:     {checkpoint_path}")
    print(f"{'='*100}\n")
    
    # 1. 加载源参数
    try:
        actor_data = load_actor_params(checkpoint_path, save_format=save_format)
        print(f"✓ Loaded checkpoint from step: {actor_data.get('step', 'unknown')}\n")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        raise
    
    # 2. 创建新agent（随机初始化）
    print("Creating new transformer agent with random initialization...")
    agent = agent_class.create(
        seed=seed,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=config,
    )
    print("✓ Agent created successfully\n")

    # print(example_batch['observations'].shape)
    # print(example_batch['actions'].shape)
    
    # 3. 检查是否有actor_onestep_flow参数
    if 'actor_onestep_flow' not in actor_data:
        raise ValueError("Checkpoint does not contain 'actor_onestep_flow' parameters")
    
    source_params = actor_data['actor_onestep_flow']
    print(f"Source modules found: {list(source_params.keys())}")
    
    # 4. 获取目标网络参数
    network_params = dict(agent.network.params)
    
    if 'modules_actor_transformer' not in network_params:
        raise ValueError("Target agent does not have 'modules_actor_transformer'")
    
    target_params = dict(network_params['modules_actor_transformer'])
    print(f"Target modules found: {list(target_params.keys())}")
    
    # 5. 分析Transformer结构，识别自动命名的组件
    component_map = analyze_transformer_structure(target_params)
    
    # 6. 参数映射 - 使用详细记录
    mapping_records = []
    
    print(f"\n{'='*100}")
    print("STARTING PARAMETER MAPPING")
    print(f"{'='*100}")
    
    # 映射 obs_encoder (2-layer MLP)
    if 'obs_encoder' in source_params and 'obs_encoder' in component_map:
        target_params, mapping_records = map_sequential_params(
            source_params['obs_encoder'],
            target_params,
            'actor_onestep_flow.obs_encoder',
            'actor_transformer.obs_encoder',
            component_map['obs_encoder'],
            mapping_records
        )
    else:
        print("\n  ⚠ WARNING: obs_encoder not found or could not be identified")
    
    # 映射 action_proj (single Dense layer)
    if 'action_proj' in source_params and 'action_proj' in target_params:
        target_params['action_proj'], mapping_records = map_embedding_params(
            source_params['action_proj'],
            target_params['action_proj'],
            'actor_onestep_flow.action_proj',
            'actor_transformer.action_proj',
            mapping_records
        )
    else:
        print("\n  ⚠ WARNING: action_proj not found in source or target")
    
    # 映射 time_embedding (3-layer MLP)
    if 'time_embedding' in source_params and 'time_embedding' in component_map:
        target_params, mapping_records = map_sequential_params(
            source_params['time_embedding'],
            target_params,
            'actor_onestep_flow.time_embedding',
            'actor_transformer.time_embedding',
            component_map['time_embedding'],
            mapping_records
        )
    else:
        print("\n  ⚠ WARNING: time_embedding not found or could not be identified")
    
    # 映射 output_mlp -> velocity_head (MLP network)
    if 'output_mlp' in source_params and 'velocity_head' in component_map:
        velocity_head_key = component_map['velocity_head']
        print(f"\n  {'='*70}")
        print(f"  Mapping: actor_onestep_flow.output_mlp → actor_transformer.{velocity_head_key}")
        print(f"  {'='*70}")
        
        target_params[velocity_head_key], mapping_records = recursive_map_with_details(
            target_params[velocity_head_key],
            source_params['output_mlp'],
            'actor_onestep_flow.output_mlp',
            f'actor_transformer.{velocity_head_key}',
            mapping_records
        )
    else:
        print("\n  ⚠ WARNING: output_mlp or velocity_head not found or could not be identified")
    
    # 7. 打印详细的映射摘要表格
    print_mapping_summary(mapping_records)
    
    # 8. 更新网络参数
    network_params['modules_actor_transformer'] = target_params
    new_network = agent.network.replace(params=network_params)
    agent = agent.replace(network=new_network)
    
    # 9. 总结
    success_count = sum(1 for r in mapping_records if r['status'] == '✓')
    fail_count = sum(1 for r in mapping_records if '✗' in r['status'])
    
    print(f"{'='*100}")
    print("CROSS-ARCHITECTURE LOADING COMPLETED")
    print(f"{'='*100}")
    print(f"\n📊 Mapping Statistics:")
    print(f"  ✓ Successfully mapped: {success_count} parameters")
    if fail_count > 0:
        print(f"  ✗ Failed to map:      {fail_count} parameters (check shape mismatches above)")
    
    print(f"\n📦 Component Mapping:")
    if 'obs_encoder' in component_map:
        print(f"  ✓ obs_encoder:     ActorVectorFieldEmb.obs_encoder → Transformer.{component_map['obs_encoder']}")
    if 'action_proj' in target_params:
        print(f"  ✓ action_proj:     ActorVectorFieldEmb.action_proj → Transformer.action_proj")
    if 'time_embedding' in component_map:
        print(f"  ✓ time_embedding:  ActorVectorFieldEmb.time_embedding → Transformer.{component_map['time_embedding']}")
    if 'velocity_head' in component_map:
        print(f"  ✓ velocity_head:   ActorVectorFieldEmb.output_mlp → Transformer.{component_map['velocity_head']}")
    
    print(f"\n🔧 Components NOT Mapped (Randomly Initialized):")
    print(f"  • Transformer decoder layers:")
    transformer_layers = [k for k in target_params.keys() if k.startswith('layer_')]
    if transformer_layers:
        print(f"    - {', '.join(transformer_layers)} (self-attention, cross-attention, feed-forward)")
    print(f"  • Critic networks:")
    print(f"    - critic, target_critic (Q-value estimation)")
    
    if hasattr(agent, 'alpha_state') and agent.alpha_state is not None:
        print(f"  • SAC alpha coefficient (entropy regularization)")
    
    print(f"\n{'='*100}\n")
    
    if fail_count > 0:
        print(f"⚠ WARNING: {fail_count} parameters failed to map. Please check the shape mismatches above.")
        print("This may indicate architecture incompatibility.\n")
    
    if success_count == 0:
        raise ValueError("No parameters were successfully mapped! Check architecture compatibility.")
    
    return agent


def load_and_create_agent_with_pretrained_actor(checkpoint_path, agent_class, seed, 
                                                example_batch, config, save_format='flax',
                                                cross_architecture=False):
    """
    加载预训练的actor参数并创建agent
    
    Args:
        checkpoint_path: actor checkpoint文件路径
        agent_class: agent类
        seed: 随机种子
        example_batch: 示例批次
        config: 配置
        save_format: checkpoint保存格式
        cross_architecture: 是否进行跨架构加载（ActorVectorFieldEmb -> Transformer）
    """
    if cross_architecture:
        return load_cross_architecture_actor(
            checkpoint_path, agent_class, seed, example_batch, config, save_format
        )
    
    # 原有的同架构加载逻辑
    print(f"\n{'='*80}")
    print(f"Loading pretrained actor from: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    try:
        actor_data = load_actor_params(checkpoint_path, save_format=save_format)
        print(f"Successfully loaded actor checkpoint from step: {actor_data.get('step', 'unknown')}")
    except Exception as e:
        print(f"Error loading actor checkpoint: {e}")
        raise
    
    print("Creating new agent with random initialization...")
    agent = agent_class.create(
        seed=seed,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=config,
    )
    print("Agent created successfully")

    print("Recursively replacing actor parameters with pretrained weights...")
    network_params = dict(agent.network.params)
    
    # print(network_params)
    # print(actor_data.items())

    all_updated_paths = []
    for key, params in actor_data.items():
        if key not in ['step', 'config', 'agent_type']:
            # if "low_dim" in FLAGS.env_name:
            #     module_key = f'{key}'
            # else:
            module_key = f'modules_{key}'
            if module_key in network_params:
                print(f"\n  Processing module: {module_key}")
                network_params[module_key], updated_paths = recursive_update_params(
                    network_params[module_key], 
                    params,
                    module_key
                )
                all_updated_paths.extend(updated_paths)
            else:
                print(f"  ⚠ Warning: {module_key} not found in agent network")
    
    if not all_updated_paths:
        raise ValueError("No actor parameters were updated! Check checkpoint compatibility.")
    
    new_network = agent.network.replace(params=network_params)
    agent = agent.replace(network=new_network)
    
    print(f"\n{'='*80}")
    print(f"Successfully loaded pretrained actor.")
    print(f"Total leaf parameters updated: {len(all_updated_paths)}")
    print(f"Critic remains randomly initialized for training.")
    print(f"{'='*80}\n")
    
    return agent


def load_and_concatenate_datasets(env, dataset_paths, num_datasets, process_train_dataset_fn, verbose=True):
    """
    从OGBench数据集路径中随机加载多个数据集并拼接
    
    Args:
        env: 环境实例
        dataset_paths: 所有可用的数据集路径列表
        num_datasets: 要加载的数据集数量，-1表示只用第一个，0表示全部，>0表示随机采样
        process_train_dataset_fn: 处理数据集的函数
        verbose: 是否打印详细信息
    
    Returns:
        concatenated_dataset: Dataset对象，包含拼接后的数据
    """
    if num_datasets == -1:
        # 默认行为：只返回None，调用方使用已加载的train_dataset
        return None
    
    # 确定要加载的数据集
    if num_datasets == 0:
        # 加载所有数据集
        selected_paths = dataset_paths
        if verbose:
            print(f"\n{'='*80}")
            print(f"Loading ALL {len(selected_paths)} OGBench datasets for replay buffer")
            print(f"{'='*80}\n")
    else:
        # 随机采样指定数量的数据集
        num_to_sample = min(num_datasets, len(dataset_paths))
        selected_paths = random.sample(dataset_paths, num_to_sample)
        if verbose:
            print(f"\n{'='*80}")
            print(f"Randomly sampling {num_to_sample} out of {len(dataset_paths)} OGBench datasets")
            print(f"Selected indices: {[dataset_paths.index(p) for p in selected_paths]}")
            print(f"{'='*80}\n")
    
    # 加载并处理所有选中的数据集
    all_datasets = []
    for i, dataset_path in enumerate(tqdm.tqdm(selected_paths, desc="Loading datasets")):
        ds, _ = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_path,
            compact_dataset=False,
            dataset_only=True,
            cur_env=env,
        )
        ds = process_train_dataset_fn(ds)
        all_datasets.append(dict(ds))
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Loaded {i + 1}/{len(selected_paths)} datasets")
    
    # 拼接所有数据集
    if verbose:
        print(f"\nConcatenating {len(all_datasets)} datasets...")
    
    concatenated_data = {}
    for key in all_datasets[0].keys():
        concatenated_data[key] = np.concatenate([ds[key] for ds in all_datasets], axis=0)
    
    # 创建新的Dataset对象
    concatenated_dataset = Dataset.create(**concatenated_data)
    
    if verbose:
        total_size = concatenated_dataset.size
        print(f"Successfully concatenated datasets!")
        print(f"Total trajectories/transitions: {total_size}")
        print(f"Average per dataset: {total_size / len(all_datasets):.1f}")
        print(f"{'='*80}\n")
    
    return concatenated_dataset


def main(_):
    agent_name = FLAGS.agent['agent_name']
    task_name = FLAGS.env_name
    base_seed = FLAGS.seed
    exp_name = f"{agent_name}_{task_name}_{base_seed}"
    
    run = setup_wandb(project='cfp', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # 数据加载
    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) 
            if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # 设置随机种子
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # 处理训练数据集
    def process_train_dataset(ds):
        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(**{k: v[:new_size] for k, v in ds.items()})
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    
    # ============ 关键修改：检查是否需要加载预训练actor ============
    if FLAGS.skip_offline_training and FLAGS.load_actor_checkpoint:
        print(f"\n{'='*80}")
        print("SKIPPING OFFLINE TRAINING - Loading pretrained actor")
        print(f"{'='*80}\n")
        
        # 确定checkpoint格式
        if FLAGS.load_actor_checkpoint.endswith('.flax'):
            save_format = 'flax'
        elif FLAGS.load_actor_checkpoint.endswith('.pkl'):
            save_format = 'pickle'
        else:
            raise ValueError(f"Unsupported checkpoint format. Use .flax or .pkl")
        
        # 加载预训练actor并创建agent
        agent = load_and_create_agent_with_pretrained_actor(
            checkpoint_path=FLAGS.load_actor_checkpoint,
            agent_class=agent_class,
            seed=FLAGS.seed,
            example_batch=example_batch,
            config=config,
            save_format=save_format,
            cross_architecture=FLAGS.cross_architecture_load
        )
        
        offline_steps_to_run = 0
        
    else:
        # 正常创建agent
        print("Creating agent from scratch...")
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
        offline_steps_to_run = FLAGS.offline_steps
    
    # ============ 设置日志 ============
    prefixes = ["eval", "env"]
    if offline_steps_to_run > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    if FLAGS.skip_offline_training and FLAGS.critic_warmup_steps > 0:
        prefixes.append("critic_warmup")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                     for prefix in prefixes},
        wandb_logger=wandb,
    )

    # ============ Offline训练（如果需要）============
    offline_init_time = time.time()
    
    if offline_steps_to_run > 0:
        print(f"\nStarting offline training for {offline_steps_to_run} steps...")
        for i in tqdm.tqdm(range(1, offline_steps_to_run + 1)):
            log_step += 1

            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name,
                    dataset_path=dataset_paths[dataset_idx],
                    compact_dataset=False,
                    dataset_only=True,
                    cur_env=env,
                )
                train_dataset = process_train_dataset(train_dataset)

            batch = train_dataset.sample_sequence(
                config['batch_size'], 
                sequence_length=FLAGS.horizon_length, 
                discount=discount
            )
            
            agent, offline_info = agent.offline_update(batch)
            if FLAGS.use_q_in_offline:
                agent, offline_info = agent.online_update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)
            
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, log_step)

            if i == offline_steps_to_run - 1 or \
                (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
                eval_info, _, _ = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                logger.log(eval_info, "eval", step=log_step)

        print(f"\nOffline training finished after {log_step} steps.")
        
        save_actor_if_enabled(agent, FLAGS.save_dir, log_step, "offline_final")
    
    # ============ Critic Warmup阶段（如果跳过offline训练）============
    if FLAGS.skip_offline_training and FLAGS.critic_warmup_steps > 0:
        print(f"\n{'='*80}")
        print(f"Starting Critic Warmup for {FLAGS.critic_warmup_steps} steps")
        print("(Actor parameters frozen, only Critic will be updated)")
        print(f"{'='*80}\n")
        
        warmup_init_time = time.time()
        log_step = int(1e6)
        td_history = []
        
        for i in tqdm.tqdm(range(1, FLAGS.critic_warmup_steps + 1)):
            log_step += 1

            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)

                print(f"Warmup using new dataset: {dataset_paths[dataset_idx]}", flush=True)

                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name,
                    dataset_path=dataset_paths[dataset_idx],
                    compact_dataset=False,
                    dataset_only=True,
                    cur_env=env,
                )

                train_dataset = process_train_dataset(train_dataset)
            
            batch = train_dataset.sample_sequence(
                config['batch_size'], 
                sequence_length=FLAGS.horizon_length, 
                discount=discount
            )
            
            # 使用warmup_update（需要在agent中实现）
            if hasattr(agent, 'warmup_update'):
                agent, warmup_info = agent.warmup_update(batch)
            else:
                # 如果没有warmup_update，使用online_update
                agent, warmup_info = agent.online_update(batch)
            
            if i % 500 == 0: 

                current_td = warmup_info.get('critic/td_error_abs', warmup_info.get('td_error_abs', 0))
                td_history.append(current_td)
                if len(td_history) > 20: 
                    td_history.pop(0)
                    td_smooth_std = np.std(td_history)
                    warmup_info['critic/td_smooth_std'] = td_smooth_std 
            
                logger.log(warmup_info, "critic_warmup", step=log_step)
            if i % (FLAGS.critic_warmup_steps // 4) == 0 or i == FLAGS.critic_warmup_steps:
                print(f"Warmup progress: {i}/{FLAGS.critic_warmup_steps}")

            

        eval_info, _, _ = evaluate(
            agent=agent,
            env=eval_env,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        logger.log(eval_info, "eval", step=log_step)
        warmup_time = time.time() - warmup_init_time
        print(f"\nCritic warmup completed in {warmup_time:.2f} seconds")
        print(f"Total steps so far: {log_step}")

    # ============ 准备Online训练 ============
    # 如果跳过了offline训练且使用OGBench数据集，并且指定了加载多个数据集
    if (FLAGS.skip_offline_training and 
        FLAGS.ogbench_dataset_dir is not None and 
        FLAGS.num_datasets_for_buffer != -1):
        
        print(f"Detected skip_offline_training with num_datasets_for_buffer={FLAGS.num_datasets_for_buffer}")
        print("Loading multiple OGBench datasets for replay buffer initialization...")
        
        concatenated_dataset = load_and_concatenate_datasets(
            env=env,
            dataset_paths=dataset_paths,
            num_datasets=FLAGS.num_datasets_for_buffer,
            process_train_dataset_fn=process_train_dataset,
            verbose=True
        )
        
        if concatenated_dataset is not None:
            # 使用拼接后的数据集
            train_dataset = concatenated_dataset
            print(f"Using concatenated dataset with size: {train_dataset.size}")
        else:
            print(f"Using original train_dataset with size: {train_dataset.size}")
    
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    print(f"Replay buffer initialized with size: {FLAGS.buffer_size}, initial data: {train_dataset.size}")
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    update_info = {}
    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()

    
    # ============ Online训练 ============
    print(f"\n{'='*80}")
    print(f"Starting Online Training for {FLAGS.online_steps} steps")
    print(f"{'='*80}\n")
    
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # Action chunking执行
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for action_item in action_chunk:
                action_queue.append(action_item)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        logger.log(env_info, "env", step=log_step)

        # 奖励调整
        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        if done:
            ob, _ = env.reset()
            action_queue = []
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(
                config['batch_size'] * FLAGS.utd_ratio, 
                sequence_length=FLAGS.horizon_length, 
                discount=discount
            )
            batch = jax.tree.map(
                lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), 
                batch
            )
            for count in range(FLAGS.more_critic-1):
                agent, update_info["online_agent"] = agent.critic_batch_update(batch)
            agent, update_info["online_agent"] = agent.online_batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)
    if FLAGS.online_steps > 0:
        save_actor_if_enabled(
            agent,
            FLAGS.save_dir,
            log_step,
            "online_final",
    )

        print(f"Online critic saved: {online_critic_path}")
    end_time = time.time()

    # ============ 清理和保存 ============
    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {
            "steps": np.array(data["steps"]),
            "qpos": np.stack(data["qpos"], axis=0), 
            "qvel": np.stack(data["qvel"], axis=0), 
            "obs": np.stack(data["obs"], axis=0), 
            "offline_time": online_init_time - offline_init_time,
            "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    if run is not None:
        with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
            f.write(run.url)
    
    print(f"\n{'='*80}")
    print("Training completed successfully!")
    print(f"Total steps: {log_step}")
    print(f"Results saved to: {FLAGS.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    app.run(main)