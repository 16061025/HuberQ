from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *


class VanillarhoFixOverlapDQN(VanillaDQN):
    '''
    Implementation of Vanilla DQN with only replay buffer (no target network)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.K = cfg['agent']['networks_num']  # number of networks
        self.Ratio_overlap = cfg['agent']['Ratio_overlap']
        if cfg['agent']['M'] is None:
            self.M = math.ceil((1 + self.Ratio_overlap) * self.K / 2)
        else:
            self.M = cfg['agent']['M']
        # Create k different: Q value network and Optimizer
        self.Q_net = [None] * self.K
        self.optimizer = [None] * self.K
        for i in range(self.K):
            self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
                                                                               **cfg['optimizer']['kwargs'])

    # def __init__(self, cfg):
    #     super().__init__(cfg)
    #     self.K = cfg['agent']['networks_num']  # number of networks
    #     self.Ratio_overlap = cfg['agent']['Ratio_overlap']
    #     if cfg['agent']['M'] is None:
    #         self.M = math.ceil((1 + self.Ratio_overlap) * self.K / 2)
    #     else:
    #         self.M = cfg['agent']['M']
    #     # Create k different: Q value network and Optimizer
    #     self.Q_net = [None] * self.K
    #     self.optimizer = [None] * self.K
    #     self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
    #     self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(),
    #                                                                        **cfg['optimizer']['kwargs'])
    #     for i in range(1, self.K):
    #         self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
    #         self.Q_net[i].load_state_dict(self.Q_net[0].state_dict())
    #         self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
    #                                                                            **cfg['optimizer']['kwargs'])


    def get_sel_est_indices(self):
        if self.M *2 < self.K:
            if np.random.random() >= 0.5:
                sel_indices =np.random.choice(np.arange(0, int(self.K/2)), self.M, replace=False)
                est_indices = np.random.choice (np.arange(int(self.K/2), self.K), self.M, replace=False)
            else:
                est_indices = np.random.choice(np.arange(0, int(self.K / 2)), self.M, replace=False)
                sel_indices = np.random.choice(np.arange(int(self.K / 2), self.K), self.M, replace=False)

        else:
            if np.random.random() >= 0.5:
                sel_indices = np.arange(self.M)
                est_indices = np.arange(start=self.K - self.M, stop=self.K)
            else:
                est_indices = np.arange(self.M)
                sel_indices = np.arange(start=self.K - self.M, stop=self.K)
        self.sel_indices = sel_indices
        self.est_indices = est_indices
        return sel_indices, est_indices

    def update_target_net(self):
        pass


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
        sel_indices, est_indices = self.get_sel_est_indices()

        with torch.no_grad():
            q_target = self.compute_q_target(batch) #batch*sel
            q_target = q_target[:, 0].squeeze() #(batch, )

        action = batch.action.long().unsqueeze(1)

        for i in self.sel_indices:
            q = self.Q_net[i](batch.state).gather(1, action).squeeze()

            loss = self.loss(q, q_target)
            # Take an optimization step
            self.optimizer[i].zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
            self.optimizer[i].step()
            if self.show_tb:
                self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

    # def learn(self):
    #     mode = 'Train'
    #     batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    #     sel_indices, est_indices = self.get_sel_est_indices()
    #     q, q_target = self.compute_q(batch), self.compute_q_target(batch)
    #
    #     # Compute loss
    #     loss = self.loss(q, q_target)
    #     # Take an optimization step
    #     for i in sel_indices:
    #         self.optimizer[i].zero_grad()
    #     loss.backward()
    #     if self.gradient_clip > 0:
    #         for i in sel_indices:
    #             nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
    #     for i in sel_indices:
    #         self.optimizer[i].step()
    #     if self.show_tb:
    #         self.logger.add_scalar(f'Loss', loss.item(), self.step_count)


    def compute_q_target(self, batch):
        with torch.no_grad():
            sel_Q_nets = self.Q_net[self.sel_indices[0]:self.sel_indices[-1] + 1]
            q_next_sAlla_list = []
            for q_net in sel_Q_nets:
                q_next_sa = q_net(batch.next_state)
                q_next_sAlla_list.append(q_next_sa)

            q_next_sAlla = torch.stack(q_next_sAlla_list, dim=-1)  # batch*N_action*N_sel
            average_q_next_sAlla = q_next_sAlla.mean(dim=-1)  # batch*N_action
            optimal_action = torch.argmax(average_q_next_sAlla, dim=1)  # batch


            est_Q_nets = self.Q_net[self.est_indices[0]:self.est_indices[-1] + 1]
            q_next_value_list = []
            for q_net in est_Q_nets:
                q_next = q_net(batch.next_state).gather(1, optimal_action.unsqueeze(-1)).squeeze()  # batch

                q_next_value_list.append(q_next)

            M_q_next = torch.stack(q_next_value_list, dim=0)  # N_est*batch
            average_q_next = M_q_next.mean(dim=0).squeeze()  # batch

            q_target = batch.reward + self.discount * average_q_next * batch.mask
            q_target = q_target.unsqueeze(-1).expand(-1, self.M)  # batch*N_sel
        return q_target

    def compute_q(self, batch):
        # Convert actions to long so they can be used as indexes
        action = batch.action.long()
        action = action.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.M)
        sel_Q_nets = self.Q_net[self.sel_indices[0]:self.sel_indices[-1] + 1]

        q_sAlla_list = []
        for q_net in sel_Q_nets:
            q_sa = q_net(batch.state)
            q_sAlla_list.append(q_sa)

        q_sAlla = torch.stack(q_sAlla_list, dim=-1)  # batch*N_action*N_sel
        q = q_sAlla.gather(1, action).squeeze(dim=1)  # batch*N_sel
        return q


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for i in range(self.K):
            q_values = self.Q_net[i](state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values





