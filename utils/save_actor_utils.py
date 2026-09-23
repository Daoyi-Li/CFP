import os
import pickle
import jax.numpy as jnp
from flax import serialization
import json

def save_actor_params(agent, save_dir, step, save_format='flax'):
    """
    保存Actor模型参数 - 支持多种Agent类型
    
    Args:
        agent: ACFQL agent对象
        save_dir: 保存目录
        step: 当前训练步数
        save_format: 保存格式，'flax'或'pickle'
    """
    # 创建actor专用保存目录
    actor_save_dir = os.path.join(save_dir, 'actor_checkpoints')
    os.makedirs(actor_save_dir, exist_ok=True)
    
    # 初始化actor数据字典
    actor_data = {
        'step': step,
        'config': dict(agent.config),  # 保存配置以便后续加载
        'agent_type': agent.config.get('agent_name', 'unknown')  # 记录agent类型
    }
    
    # 自动检测并保存所有actor相关的网络参数
    network_params = agent.network.params
    actor_networks = []
    
    for key in network_params.keys():
        if key.startswith('modules_actor'):
            actor_networks.append(key)
            # 移除'modules_'前缀作为保存的key
            save_key = key.replace('modules_', '')
            actor_data[save_key] = network_params[key]
    
    print(f"Found actor networks: {actor_networks}")
    
    # 如果没有找到任何actor网络，尝试常见的名称
    if not actor_networks:
        common_actor_keys = [
            'modules_actor_onestep_flow',
            'modules_actor_bc_flow',
            'modules_actor_transformer', 
            'modules_actor',
            'modules_policy'
        ]
        for key in common_actor_keys:
            if key in network_params:
                actor_networks.append(key)
                save_key = key.replace('modules_', '')
                actor_data[save_key] = network_params[key]
                break
    
    if save_format == 'flax':
        # 使用Flax序列化
        filename = f'actor_step_{step}.flax'
        filepath = os.path.join(actor_save_dir, filename)
        
        with open(filepath, 'wb') as f:
            serialized_data = serialization.to_bytes(actor_data)
            f.write(serialized_data)
            
    elif save_format == 'pickle':
        # 使用pickle保存
        filename = f'actor_step_{step}.pkl'
        filepath = os.path.join(actor_save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(actor_data, f)
    
    # 保存一个最新的检查点链接
    latest_link = os.path.join(actor_save_dir, 'latest_actor.txt')
    with open(latest_link, 'w') as f:
        f.write(filename)
    
    print(f"Actor parameters saved to: {filepath}")
    return filepath

def load_actor_params(filepath, save_format='flax'):
    """
    加载Actor模型参数
    
    Args:
        filepath: 保存文件路径
        save_format: 保存格式，'flax'或'pickle'
    
    Returns:
        actor_data: 包含actor参数和配置的字典
    """
    if save_format == 'flax':
        with open(filepath, 'rb') as f:
            serialized_data = f.read()
            actor_data = serialization.from_bytes(target=None, encoded_bytes=serialized_data)
    elif save_format == 'pickle':
        with open(filepath, 'rb') as f:
            actor_data = pickle.load(f)
    
    return actor_data

def save_actor_params_lightweight(agent, save_dir, step):
    """
    轻量级保存，只保存参数数组，不保存配置 - 支持多种Agent类型
    """
    actor_save_dir = os.path.join(save_dir, 'actor_checkpoints')
    os.makedirs(actor_save_dir, exist_ok=True)
    
    filename = f'actor_params_step_{step}.npz'
    filepath = os.path.join(actor_save_dir, filename)
    
    # 将参数展平为numpy数组保存
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # 自动检测并保存所有actor相关的网络参数
    network_params = agent.network.params
    all_flat_params = {}
    
    for key in network_params.keys():
        if key.startswith('modules_actor'):
            actor_params = network_params[key]
            # 为每个actor网络创建带前缀的参数
            save_key = key.replace('modules_', '')
            flat_params = flatten_dict(actor_params, parent_key=save_key)
            all_flat_params.update(flat_params)
    
    # 如果没有找到actor网络，尝试常见名称
    if not all_flat_params:
        common_keys = [
            'modules_actor_onestep_flow', 
            'modules_actor_bc_flow', 
            'modules_actor_transformer',
            'modules_actor', 
            'modules_policy'
        ]
        for key in common_keys:
            if key in network_params:
                actor_params = network_params[key]
                save_key = key.replace('modules_', '')
                flat_params = flatten_dict(actor_params, parent_key=save_key)
                all_flat_params.update(flat_params)
                break
    
    # 转换为numpy格式保存
    numpy_params = {k: jnp.asarray(v) for k, v in all_flat_params.items()}
    numpy_params['step'] = step
    numpy_params['agent_type'] = agent.config.get('agent_name', 'unknown')
    
    jnp.savez(filepath, **numpy_params)
    
    print(f"Lightweight actor parameters saved to: {filepath}")
    return filepath