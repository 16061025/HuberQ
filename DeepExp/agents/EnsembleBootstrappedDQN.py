

from agents.VanillaDQN import *
from agents.DDQN import *


class EnsembleBootstrappedDQN(DDQN):
    def __init__(self, cfg):
        super().__init__(cfg)

# class EnsembleBootstrappedDQN(VanillaDQN):
#     '''
#     Implementation of Vanilla DQN with only replay buffer (no target network)
#     '''
# 
#     def __init__(self, cfg):
#         super().__init__(cfg)
# 
#         self.K = cfg['agent']['target_networks_num']  # number of target networks
#         # Create k different: Q value network, Target Q value network and Optimizer
#         self.Q_net = [None] * self.K
#         self.Q_net_target = [None] * self.K
# 
#         self.optimizer = [None] * self.K
# 
#         for i in range(self.K):
#             self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
#             self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
#             self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
#                                                                                **cfg['optimizer']['kwargs'])
#             self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
#             self.Q_net_target[i].eval()
# 
# 
# 
#     def update_target_net(self):
#         if self.step_count % self.cfg['target_network_update_steps'] == 0:
#             for i in range(self.K):
#                 self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
# 
# 
#     def learn(self):
#         mode = 'Train'
#         batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
# 
#         indices = np.arange(self.K)
#         random_idx = np.random.choice(self.K)
#         self.selector_indices = indices[random_idx]
#         self.estimator_indices = np.delete(indices, random_idx)
# 
# 
#         q, q_target = self.compute_q(batch), self.compute_q_target(batch)
# 
#         loss = self.loss(q, q_target)
#         # Take an optimization step
#         self.optimizer[self.selector_indices].zero_grad()
#         loss.backward()
#         if self.gradient_clip > 0:
#             nn.utils.clip_grad_norm_(self.Q_net[self.selector_indices].parameters(), self.gradient_clip)
#         self.optimizer[self.selector_indices].step()
#         if self.show_tb:
#             self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
# 
# 
# 
#     def compute_q_target(self, batch):
#         with torch.no_grad():
#             q_ensemble = 0
#             for i in self.estimator_indices:
#                 q = self.Q_net_target[i](batch.next_state)
#                 q_ensemble = q_ensemble + q
#             q_next = q_ensemble.max(1)[0] / (self.K-1)
#             q_target = batch.reward + self.discount * q_next * batch.mask
# 
#         return q_target
# 
#     def compute_q(self, batch):
# 
#         action = batch.action.long().unsqueeze(1)
#         q = self.Q_net[self.selector_indices](batch.state).gather(1, action).squeeze()
# 
#         return q
# 
# 
#     def get_action_selection_q_values(self, state):
#         q_ensemble = self.Q_net[0](state)
#         for i in range(1, self.K):
#             q = self.Q_net[i](state)
#             q_ensemble = q_ensemble + q
#         q_ensemble = to_numpy(q_ensemble / self.K).flatten()
#         return q_ensemble
