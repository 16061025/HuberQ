from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *
import matplotlib.pyplot as plt


class Target2DDQN(VanillaDQN):
    '''
    Implementation of Vanilla DQN with only replay buffer (no target network)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.Q_net = [None]*2
        self.optimizer = [None]*2
        for i in range(2):
            self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(),
                                                                           **cfg['optimizer']['kwargs'])

        self.Q_net_target = [None]*2
        for i in range(2):
            self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
            # Load target Q value network
            self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
            self.Q_net_target[i].eval()


        self.A = 0
        self.B = 1

        self.sel = cfg['agent']['sel']
        #self.imshowpara()
        self.updateAcount = 0
        self.updateBcount = 0


    def update_target_net(self):
        if self.step_count % self.cfg['target_network_update_steps'] == 0:
            for i in range(2):
                self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        prob = np.random.random()
        if prob >= 0.5:
            action = batch.action.long().unsqueeze(1)
            q = self.Q_net[self.A](batch.state).gather(1, action).squeeze()

            with torch.no_grad():
                if self.sel == "Q":
                    best_actions = self.Q_net[self.A](batch.next_state).argmax(1).unsqueeze(1)
                else:
                    best_actions = self.Q_net_target[self.A](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net_target[self.B](batch.next_state).gather(1, best_actions).squeeze()
                q_target = batch.reward + self.discount * q_next * batch.mask

            # Compute loss
            loss = self.loss(q, q_target)
            # Take an optimization step

            self.optimizer[self.A].zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[self.A].parameters(), self.gradient_clip)

            self.optimizer[self.A].step()
            if self.show_tb:
                self.logger.add_scalar(f'LossA', loss.item(), self.step_count)


        else:
            action = batch.action.long().unsqueeze(1)
            q = self.Q_net[self.B](batch.state).gather(1, action).squeeze()

            with torch.no_grad():
                if self.sel == "Q":
                    best_actions = self.Q_net[self.B](batch.next_state).argmax(1).unsqueeze(1)
                else:
                    best_actions = self.Q_net_target[self.B](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net_target[self.A](batch.next_state).gather(1, best_actions).squeeze()
                q_target = batch.reward + self.discount * q_next * batch.mask

            # Compute loss
            loss = self.loss(q, q_target)
            # Take an optimization step

            self.optimizer[self.B].zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[self.B].parameters(), self.gradient_clip)

            self.optimizer[self.B].step()
            if self.show_tb:
                self.logger.add_scalar(f'LossA', loss.item(), self.step_count)


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for Q_net in self.Q_net:
            q_values = Q_net(state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values
