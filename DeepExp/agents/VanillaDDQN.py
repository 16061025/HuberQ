from agents.VanillaDQN import *


class VanillaDDQN(VanillaDQN):
    '''
    Implementation of DQN with target network and replay buffer
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        # Create target Q value network
        self.Q_net = [None] * 2
        self.optimizer = [None] * 2
        self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(),
                                                                           **cfg['optimizer']['kwargs'])

        self.Q_net[1] = self.createNN(cfg['env']['input_type']).to(self.device)
        self.optimizer[1] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[1].parameters(),
                                                                           **cfg['optimizer']['kwargs'])

    def update_target_net(self):
        pass

    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        if np.random.random() > 0.5:
            action = batch.action.long().unsqueeze(1)
            q_A = self.Q_net[0](batch.state).gather(1, action).squeeze()

            with torch.no_grad():
                best_actions_A = self.Q_net[0](batch.next_state).argmax(1).unsqueeze(1)
                q_nextA = self.Q_net[1](batch.next_state).gather(1, best_actions_A).squeeze()
                q_target_A = batch.reward + self.discount * q_nextA * batch.mask

            loss = self.loss(q_A, q_target_A)
            # Take an optimization step
            self.optimizer[0].zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[0].parameters(), self.gradient_clip)

            self.optimizer[0].step()

        else:
            action = batch.action.long().unsqueeze(1)
            q_B = self.Q_net[1](batch.state).gather(1, action).squeeze()

            with torch.no_grad():
                best_actions_B = self.Q_net[1](batch.next_state).argmax(1).unsqueeze(1)
                q_nextB = self.Q_net[0](batch.next_state).gather(1, best_actions_B).squeeze()
                q_target_B = batch.reward + self.discount * q_nextB * batch.mask

            loss = self.loss(q_B, q_target_B)
            # Take an optimization step
            self.optimizer[1].zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[1].parameters(), self.gradient_clip)

            self.optimizer[1].step()



    def get_action_selection_q_values(self, state):
        q_values_list = []
        for Q_net in self.Q_net:
            q_values = Q_net(state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()
        return q_values