import numpy as np

from agents import DQN
from agents.VanillaDQN import *
from agents.MaxminDQN import *

class AdaptiveOrderDQN(MaxminDQN):
    def __init__(self, cfg):
        cfg['agent']['target_networks_num'] = cfg['agent']['M']
        super().__init__(cfg)

# class AdaptiveOrderDQN(VanillaDQN):
#     '''
#     Implementation of DQN with target network and replay buffer
#     '''
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         # Create target Q value network
#         self.C = cfg['agent']['C']
#         self.m = cfg['agent']['m']
#         self.M = cfg['agent']['M']
#         self.AdaptiveOrder = cfg['agent']['AdaptiveOrder']
#
#         self.Q_net = [None] * self.M
#         self.Q_net_target = [None] * self.M
#         self.optimizer = [None] * self.M
#         for i in range(self.M):
#             self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
#             #torch.save(self.Q_net[i].state_dict(), f'../Q{i:d}para.pth')
#             #self.Q_net[i].load_state_dict(torch.load(f'Q{i:d}para.pth'))
#             self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
#             self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
#                                                                                **cfg['optimizer']['kwargs'])
#             self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
#             self.Q_net_target[i].eval()
#
#     def update_target_net(self):
#         if self.step_count % self.cfg['target_network_update_steps'] == 0:
#             for i in range(self.M):
#                 self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
#
#     def compute_q_target(self, batch):
#         with (torch.no_grad()):
#             allQNet_q_next = []
#             for i in range(self.M):
#                 q_next = self.Q_net_target[i](batch.next_state)
#                 allQNet_q_next.append(q_next)
#             allQNet_q_next = torch.stack(allQNet_q_next, dim=1)
#             #mean_allQNet_q_next = torch.mean(allQNet_q_next, dim=1).squeeze()
#
#             q_next = torch.kthvalue(allQNet_q_next, k=self.m, dim=1).values
#             q_next = q_next.max(dim=1).values
#
#             q_target = batch.reward + self.discount * q_next * batch.mask
#
#         return q_target
#
#     def compute_q(self, batch):
#         # Convert actions to long so they can be used as indexes
#         action = batch.action.long().unsqueeze(1)
#
#         self.update_Q_net_index = np.random.choice(self.M)
#         q = self.Q_net[self.update_Q_net_index](batch.state).gather(1, action).squeeze()
#         return q
#
#     def learn(self):
#         mode = 'Train'
#         batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
#         q, q_target = self.compute_q(batch), self.compute_q_target(batch)
#
#         # Compute loss
#         loss = self.loss(q, q_target)
#         # Take an optimization step
#         self.optimizer[self.update_Q_net_index].zero_grad()
#         loss.backward()
#         if self.gradient_clip > 0:
#             nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)
#         self.optimizer[self.update_Q_net_index].step()
#         if self.show_tb:
#             self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
#
#     def get_action_selection_q_values(self, state):
#         q_values_list = []
#         for Q_net in self.Q_net:
#             q_values = Q_net(state)
#             q_values_list.append(q_values)
#         q_values = torch.cat(q_values_list, dim=0)
#         q_values = torch.kthvalue(q_values, k=self.m, dim=0).values
#         q_values = to_numpy(q_values).flatten()
#
#
#         return q_values
