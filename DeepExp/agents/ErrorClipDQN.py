from agents.VanillaDQN import *


class ErrorClipDQN(VanillaDQN):
    '''
    Implementation of DQN with target network and replay buffer
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        # Create target Q value network
        self.Q_net_target = [None]
        self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        # Load target Q value network
        self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
        self.Q_net_target[0].eval()
        self.delta = cfg['agent']['delta']

    def update_target_net(self):
        if self.step_count % self.cfg['target_network_update_steps'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
        q, q_target = self.compute_q(batch), self.compute_q_target(batch)
        q_target = torch.min(q.clone().detach()+self.delta, q_target)


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
            q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
            q_target = batch.reward + self.discount * q_next * batch.mask
        return q_target