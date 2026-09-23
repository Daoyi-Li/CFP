from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acfql_gru import ACFQLAgent_GRU
from agents.acfql_gru_offline import ACFQLAgent_GRUOffline
from agents.acfql_offline import ACFQLAgentOffline
from agents.acfql_gru_ablation_online import ACFQLAgent_GRUAblationOnline
from agents.acfql_transformer_ablation_online import ACFQLAgent_TransformerAblationOnline
from agents.acfql_ablation_online import ACFQLAgent_AblationOnline
from agents.acfql_gru_ablation_online_sac import ACFQLAgent_GRUAblationOnlineSAC
from agents.acfql_gru_crossq import ACFQLAgent_CrossQ
from agents.acfql_transformer_ablation_online_sac import ACFQLAgent_TransformerAblationOnlineSAC
from agents.acfql_gru_finetune_sac import ACFQLAgent_GRUFinetuneSAC
from agents.acfql_transformer_finetune_sac import ACFQLAgent_TransformerFinetuneSAC
from agents.acfql_finetune import ACFQLAgent_Finetune
from agents.acfql_finetune_montecarlo import ACFQLAgent_Finetune_MonteCarlo
agents = dict(
    acfql=ACFQLAgent,
    acfql_finetune = ACFQLAgent_Finetune,
    acfql_finetune_montecarlo = ACFQLAgent_Finetune_MonteCarlo,
    acrlpd=ACRLPDAgent,
    acfql_gru=ACFQLAgent_GRU,
    acfql_gru_offline=ACFQLAgent_GRUOffline,
    acfql_offline = ACFQLAgentOffline,
    acfql_gru_ablation_online = ACFQLAgent_GRUAblationOnline,
    acfql_transformer_ablation_online = ACFQLAgent_TransformerAblationOnline,
    acfql_ablation_online = ACFQLAgent_AblationOnline,
    acfql_gru_ablation_online_sac = ACFQLAgent_GRUAblationOnlineSAC,
    acfql_gru_crossq = ACFQLAgent_CrossQ,
    acfql_transformer_ablation_online_sac = ACFQLAgent_TransformerAblationOnlineSAC,
    acfql_gru_finetune_sac = ACFQLAgent_GRUFinetuneSAC,
    acfql_transformer_finetune_sac = ACFQLAgent_TransformerFinetuneSAC
)
