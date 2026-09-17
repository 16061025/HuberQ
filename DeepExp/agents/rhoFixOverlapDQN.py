from distutils.core import setup_keywords

from agents.VanillaDQN import *


class rhoFixOverlapDQN(VanillaDQN):
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
        self.Q_net = [None] * self.M
        self.Q_net_target = [None] * self.M
        self.optimizer = [None] * self.M

        for i in range(self.M):
            self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
                                                                               **cfg['optimizer']['kwargs'])
            self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
            self.Q_net_target[i].eval()
        self.N_DDQN = self.K-self.M



    def update_target_net(self):
        if self.step_count % self.cfg['target_network_update_steps'] == 0:
            a = 1
            for i in range(self.M):
                self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())



    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        q, q_target = self.compute_q(batch), self.compute_q_target(batch)
        # Compute loss
        loss = self.loss(q, q_target)
        # Take an optimization step
        for i in range(self.M):
            self.optimizer[i].zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            for i in range(self.M):
                nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
        for i in range(self.M):
            self.optimizer[i].step()
        if self.show_tb:
            self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

    def compute_q_target(self, batch):
        with torch.no_grad():
            sel_DQ_nets = self.Q_net[0:self.N_DDQN]
            sel_Q_nets_target = self.Q_net_target[self.N_DDQN:]

            q_next_sAlla_list = []
            for q_net in sel_DQ_nets + sel_Q_nets_target:
                q_next_sa = q_net(batch.next_state)
                q_next_sAlla_list.append(q_next_sa)

            q_sAlla = torch.stack(q_next_sAlla_list, dim=-1)  # batch*N_action*N_sel
            average_q_sAlla = q_sAlla.mean(dim=-1)  # batch*N_action
            optimal_action = torch.argmax(average_q_sAlla, dim=1)  # batch

            q_next_value_list = []
            for q_net in self.Q_net_target:
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

        q_sAlla_list = []
        for q_net in self.Q_net:
            q_sa = q_net(batch.state)
            q_sAlla_list.append(q_sa)

        q_sAlla = torch.stack(q_sAlla_list, dim=-1)  # batch*N_action*N_sel
        q = q_sAlla.gather(1, action).squeeze(dim=1)  # batch*N_sel
        return q


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for i in range(self.M):
            q_values = self.Q_net[i](state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values

