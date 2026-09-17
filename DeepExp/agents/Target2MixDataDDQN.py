from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *
import matplotlib.pyplot as plt


class Target2MixDataDDQN(VanillaDQN):
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

        self.Mix_ratio = cfg['agent']['rho']
        self.updateA = False
        self.updateB = False
        self.updateAB = False

        self.updateAProbability = (1-self.Mix_ratio)/2
        self.updateABProbability = self.Mix_ratio
        self.A = 0
        self.B = 1



    def update_target_net(self):
        if self.step_count % self.cfg['target_network_update_steps'] == 0:
            for i in range(2):
                self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        prob = np.random.random()
        if prob >= 1 - self.Mix_ratio:
            self.updateA, self.updateB, self.updateAB = False, False, True
        elif prob >= self.updateAProbability:
            self.updateA, self.updateB, self.updateAB = True, False, False
        else:
            self.updateA, self.updateB, self.updateAB = False, True, False

        q_A, q_B = self.compute_q(batch)
        q_target_A, q_target_B = self.compute_q_target(batch)

        if q_A is not None:
            # Compute loss
            loss = self.loss(q_A, q_target_A)
            # Take an optimization step

            self.optimizer[self.A].zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[self.A].parameters(), self.gradient_clip)

            self.optimizer[self.A].step()
            if self.show_tb:
                self.logger.add_scalar(f'LossA', loss.item(), self.step_count)

        if q_B is not None:
            # Compute loss
            loss = self.loss(q_B, q_target_B)
            # Take an optimization step

            self.optimizer[self.B].zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[self.B].parameters(), self.gradient_clip)

            self.optimizer[self.B].step()

            if self.show_tb:
                self.logger.add_scalar(f'LossB', loss.item(), self.step_count)


    def compute_q_target(self, batch):
        with torch.no_grad():
            q_target_A, q_target_B = None, None
            if self.updateA:
                best_actions = self.Q_net_target[self.A](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net_target[self.B](batch.next_state).gather(1, best_actions).squeeze()
                q_target_A = batch.reward + self.discount * q_next * batch.mask

            elif self.updateB:
                best_actions = self.Q_net_target[self.B](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net_target[self.A](batch.next_state).gather(1, best_actions).squeeze()
                q_target_B = batch.reward + self.discount * q_next * batch.mask

            elif self.updateAB:
                best_actions_A = self.Q_net_target[self.A](batch.next_state).argmax(1).unsqueeze(1)
                q_nextA = self.Q_net_target[self.A](batch.next_state).gather(1, best_actions_A).squeeze()
                q_target_A = batch.reward + self.discount * q_nextA * batch.mask

                best_actions_B = self.Q_net_target[self.B](batch.next_state).argmax(1).unsqueeze(1)
                q_nextB = self.Q_net_target[self.B](batch.next_state).gather(1, best_actions_B).squeeze()
                q_target_B = batch.reward + self.discount * q_nextB * batch.mask

        return q_target_A, q_target_B

    def compute_q(self, batch):
        q_A, q_B = None, None

        if self.updateA:
            action = batch.action.long().unsqueeze(1)
            q_A = self.Q_net[self.A](batch.state).gather(1, action).squeeze()

        elif self.updateB:
            action = batch.action.long().unsqueeze(1)
            q_B = self.Q_net[self.B](batch.state).gather(1, action).squeeze()

        elif self.updateAB:
            action = batch.action.long().unsqueeze(1)
            q_A = self.Q_net[self.A](batch.state).gather(1, action).squeeze()
            q_B = self.Q_net[self.B](batch.state).gather(1, action).squeeze()

        return q_A, q_B


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for Q_net in self.Q_net:
            q_values = Q_net(state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values

    # def get_action_selection_q_values(self, state):
    #     prob = np.random.random()
    #     if prob >= 1 - self.Mix_ratio:
    #         self.updateA, self.updateB, self.updateAB = False, False, True
    #     elif prob >= self.updateAProbability:
    #         self.updateA, self.updateB, self.updateAB = True, False, False
    #     else:
    #         self.updateA, self.updateB, self.updateAB = False, True, False
    #
    #     if self.updateA:
    #         q_values = self.Q_net[self.A](state)
    #         q_values = q_values.mean(dim=0, keepdim=True)
    #         q_values = to_numpy(q_values).flatten()
    #
    #     elif self.updateB:
    #         q_values = self.Q_net[self.B](state)
    #         q_values = q_values.mean(dim=0, keepdim=True)
    #         q_values = to_numpy(q_values).flatten()
    #     elif self.updateAB:
    #         q_values_list = []
    #         for Q_net in self.Q_net:
    #             q_values = Q_net(state)
    #             q_values_list.append(q_values)
    #         q_values = torch.cat(q_values_list, dim=0)
    #         q_values = q_values.mean(dim=0, keepdim=True)
    #         q_values = to_numpy(q_values).flatten()
    #
    #     return q_values
