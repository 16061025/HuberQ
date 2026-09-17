from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *
import matplotlib.pyplot as plt


class MixDataDDQN(VanillaDQN):
    '''
    Implementation of Vanilla DQN with only replay buffer (no target network)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.Q_net = [None]
        self.optimizer = [None]
        self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(),
                                                                           **cfg['optimizer']['kwargs'])
        self.Q_net_target = [None]
        self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        # Load target Q value network
        self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
        self.Q_net_target[0].eval()

        self.Mix_ratio = cfg['agent']['rho']

        self.updateAB = False
        self.updateABProbability = self.Mix_ratio

        #self.imshowpara()
        self.updateAcount = 0
        self.updateBcount = 0


    def update_target_net(self):
        if self.step_count % self.cfg['target_network_update_steps'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        prob = np.random.random()

        if prob < self.updateABProbability:
            self.updateAB = True
        else:
            self.updateAB = False

        q, q_target = self.compute_q(batch), self.compute_q_target(batch)


        # gradients_A = {}
        # gradients_B = {}

        # Compute loss
        loss = self.loss(q, q_target)
        # Take an optimization step

        self.optimizer[self.update_Q_net_index].zero_grad()
        loss.backward()

        # for name, param in self.Q_net[0].named_parameters():
        #     if param.grad is not None:
        #         gradients_A[name] = param.grad.detach().clone()

        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)

        self.optimizer[self.update_Q_net_index].step()
        if self.show_tb:
            self.logger.add_scalar(f'LossA', loss.item(), self.step_count)



        # for name in gradients_A.keys():
        #     if name in gradients_B:
        #         grad_A = gradients_A[name]
        #         grad_B = gradients_B[name]
        #
        #         if not torch.equal(grad_A, grad_B):
        #
        #             # 计算差异指标
        #             abs_diff = torch.abs(grad_A - grad_B)
        #             max_diff = torch.max(abs_diff).item()
        #             mean_diff = torch.mean(abs_diff).item()
        #             rel_diff = torch.norm(grad_A - grad_B) / (torch.norm(grad_A) + 1e-8)
        #
        #             print(f"🔍 {name}:")
        #             print(f"   最大绝对差异: {max_diff:.6f}")
        #             print(f"   平均绝对差异: {mean_diff:.6f}")
        #             print(f"   相对差异: {rel_diff:.6f}")
        #             print(f"   形状: {grad_A.shape}")
        #
        # self.quick_parameter_diff_check(self.Q_net[0], self.Q_net[1])
        #self.imshowpara()
        # if self.updateAcount != self.updateBcount:
        #     print("AAAA")
        # self.compare_model_params()



    def compute_q_target(self, batch):
        with torch.no_grad():

            if self.updateAB:
                q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
                q_target = batch.reward + self.discount * q_next * batch.mask

            else:
                best_actions = self.Q_net[0](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net_target[0](batch.next_state).gather(1, best_actions).squeeze()
                q_target = batch.reward + self.discount * q_next * batch.mask

        return q_target


    def get_action_selection_q_values(self, state):
        q_values_list = []
        for Q_net in self.Q_net:
            q_values = Q_net(state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values

    def quick_parameter_diff_check(self, model_A, model_B, atol=1e-6):
        """
        快速检查参数差异，只显示有差异的参数名
        """
        print("参数差异快速检查:")
        print("=" * 40)

        different_params = []

        for (name_A, param_A), (name_B, param_B) in zip(
                model_A.named_parameters(), model_B.named_parameters()):

            if name_A != name_B:
                different_params.append((name_A, "参数名不匹配"))
                continue

            if param_A.shape != param_B.shape:
                different_params.append((name_A, "形状不同"))
                continue

            if not torch.allclose(param_A, param_B, atol=atol):
                max_diff = torch.max(torch.abs(param_A - param_B)).item()
                different_params.append((name_A, f"数值差异 (最大: {max_diff:.4f})"))

        if not different_params:
            print("✅ 所有参数相同")
        else:
            print(f"❌ 发现 {len(different_params)} 个参数不同:")
            for param_name, reason in different_params:
                print(f"   - {param_name}: {reason}")

    def imshowpara(self):
        Q1 = self.Q_net[self.A]
        Q2 = self.Q_net[self.B]
        p1 = np.concatenate([p.data.cpu().numpy().ravel() for p in Q1.parameters()])
        p2 = np.concatenate([p.data.cpu().numpy().ravel() for p in Q2.parameters()])

        # 重塑和绘图
        n, r = len(p1), int(np.sqrt(len(p1)))
        c = (n + r - 1) // r

        fig, axes = plt.subplots(3, 1, figsize=(6, 9))
        for ax, data, title in zip(axes, [p1, p2, p1 - p2], ['Q1', 'Q2', 'Q1-Q2']):
            mat = np.zeros(r * c)
            mat[:n] = data
            ax.imshow(mat.reshape(r, c), cmap='viridis' if 'Q' in title else 'RdBu_r')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def compare_model_params(self):
        Q1 = self.Q_net[self.A]
        Q2 = self.Q_net[self.B]
        """比较两个模型参数的平均差异"""
        total_diff = 0
        total_params = 0

        print(f"step{self.step_count}模型参数平均差异比较:")
        print("=" * 40)

        for (name1, p1), (name2, p2) in zip(Q1.named_parameters(), Q2.named_parameters()):
            # 计算该参数的平均绝对差异
            mean_diff = torch.mean(torch.abs(p1 - p2)).item()
            num_params = p1.numel()

            total_diff += mean_diff * num_params
            total_params += num_params

            #print(f"{name1}: {mean_diff:.6f}")

        # 计算总体平均差异
        overall_mean_diff = total_diff / total_params if total_params > 0 else 0
        print(f"\n总体平均差异: {overall_mean_diff:.6f}")
        print(f"总参数数量: {total_params}")