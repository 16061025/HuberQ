import math
from distutils.core import setup_keywords

import numpy as np
import torch

from agents.VanillaDQN import *
import matplotlib.pyplot as plt


class ActionMultiplexDQN(VanillaDQN):
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

        #self.multiplex_ratio = cfg['agent']['multiplex']
        #self.topK = int(round(self.multiplex_ratio * (self.action_size-1)+1, 0))
        self.topK = int(cfg['agent']['multiplex'])
        self.A = 0
        self.B = 1
        self.update_Q_net_index = self.A
        self.updateA = True
        self.updateB = False


    def update_target_net(self):
        if self.step_count % self.cfg['target_network_update_steps'] == 0:
            for i in range(2):
                self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        prob = np.random.random()
        if prob > 0.5:
            self.updateA, self.updateB = True, False
            self.update_Q_net_index = self.A
        else:
            self.updateA, self.updateB = False, True
            self.update_Q_net_index = self.B

        q= self.compute_q(batch)
        q_target= self.compute_q_target(batch)


        # Compute loss
        loss = self.loss(q, q_target)
        # Take an optimization step

        self.optimizer[self.update_Q_net_index].zero_grad()
        loss.backward()

        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)

        self.optimizer[self.update_Q_net_index].step()
        if self.show_tb:
            self.logger.add_scalar(f'LossA', loss.item(), self.step_count)




    def compute_q_target(self, batch):
        with torch.no_grad():
            q_target = None

            if self.updateA:


                best_actions = self.Q_net_target[self.A](batch.next_state).topk(self.topK)[1]
                q_next = self.Q_net_target[self.B](batch.next_state).gather(1, best_actions).max(1)[0]

                q_target = batch.reward + self.discount * q_next * batch.mask

            elif self.updateB:

                best_actions = self.Q_net_target[self.B](batch.next_state).topk(self.topK)[1]
                q_next = self.Q_net_target[self.A](batch.next_state).gather(1, best_actions).max(1)[0]

                q_target = batch.reward + self.discount * q_next * batch.mask



        return q_target

    def compute_q(self, batch):
        q= None

        if self.updateA:
            action = batch.action.long().unsqueeze(1)
            q = self.Q_net[self.A](batch.state).gather(1, action).squeeze()

        elif self.updateB:
            action = batch.action.long().unsqueeze(1)
            q = self.Q_net[self.B](batch.state).gather(1, action).squeeze()

        return q


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
