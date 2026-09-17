from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *


class WeightedDQN(VanillaDQN):
    '''
    Implementation of Vanilla DQN with only replay buffer (no target network)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.c = cfg['agent']['c']
        self.U = 0
        self.V = 1
        self.Q_net = [None] * 2
        self.optimizer = [None] * 2
        for i in range(2):
            self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
                                                                               **cfg['optimizer']['kwargs'])

    def update_target_net(self):
        pass




    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
        if np.random.random() > 0.5:
            self.updateU, self.updateV = True, False
            self.update_Q_net_index = self.U

        else:
            self.updateU, self.updateV = False, True
            self.update_Q_net_index = self.V


        q, q_target = self.compute_q(batch), self.compute_q_target(batch)

        # Compute loss
        loss = self.loss(q, q_target)
        # Take an optimization step

        self.optimizer[self.update_Q_net_index].zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)

        self.optimizer[self.update_Q_net_index].step()
        if self.show_tb:
            self.logger.add_scalar(f'Loss', loss.item(), self.step_count)


    def compute_q_target(self, batch):
        with torch.no_grad():
            if self.updateU:
                Q_U_next = self.Q_net[self.U](batch.next_state)
                optimal_a = Q_U_next.argmax(1).long().unsqueeze(1)
                a_L = Q_U_next.argmin(1).long().unsqueeze(1)
                Q_U_next_s_optimal_a = Q_U_next.gather(1, optimal_a).squeeze()

                Q_V_next = self.Q_net[self.V](batch.next_state)
                Q_V_next_s_optimal_a = Q_V_next.gather(1, optimal_a).squeeze()
                Q_V_next_s_a_L = Q_V_next.gather(1, a_L).squeeze()
                abs_err = torch.abs(Q_V_next_s_optimal_a - Q_V_next_s_a_L)
                beta_U = abs_err / (self.c + abs_err)

                q_target = batch.reward + self.discount * (beta_U*Q_U_next_s_optimal_a+(1-beta_U)*Q_V_next_s_optimal_a) * batch.mask
            else:
                Q_V_next = self.Q_net[self.V](batch.next_state)
                optimal_a = Q_V_next.argmax(1).long().unsqueeze(1)
                a_L = Q_V_next.argmin(1).long().unsqueeze(1)
                Q_V_next_s_optimal_a = Q_V_next.gather(1, optimal_a).squeeze()

                Q_U_next = self.Q_net[self.U](batch.next_state)
                Q_U_next_s_optimal_a = Q_U_next.gather(1, optimal_a).squeeze()
                Q_U_next_s_a_L = Q_U_next.gather(1, a_L).squeeze()
                abs_err = torch.abs(Q_U_next_s_optimal_a - Q_U_next_s_a_L)
                beta_V = abs_err / (self.c + abs_err)

                q_target = batch.reward + self.discount * (beta_V * Q_V_next_s_optimal_a + (1 - beta_V) * Q_U_next_s_optimal_a) * batch.mask

        return q_target

    def compute_q(self, batch):
        # Convert actions to long so they can be used as indexes
        action = batch.action.long().unsqueeze(1)
        q = self.Q_net[self.update_Q_net_index](batch.state).gather(1, action).squeeze()
        return q


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for i in range(len(self.Q_net)):
            q_values = self.Q_net[i](state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values





