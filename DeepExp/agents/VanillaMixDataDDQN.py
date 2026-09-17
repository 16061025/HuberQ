from distutils.core import setup_keywords

import torch

from agents.VanillaDQN import *
import matplotlib.pyplot as plt


class VanillaMixDataDDQN(VanillaDQN):
    '''
    Implementation of Vanilla DQN with only replay buffer (no target network)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)



        self.Q_net = [None] * 2
        self.optimizer = [None] * 2
        self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(),
                                                                           **cfg['optimizer']['kwargs'])



        self.Q_net[1] = self.createNN(cfg['env']['input_type']).to(self.device)
        self.optimizer[1] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[1].parameters(),
                                                                           **cfg['optimizer']['kwargs'])

        self.Mix_ratio = cfg['agent']['rho']
        self.updateA = False
        self.updateB = False
        self.updateAB = False
        self.updateAProbability = (1-self.Mix_ratio)/2
        self.A = 0
        self.B = 1
        #self.quick_parameter_diff_check(self.Q_net[0], self.Q_net[1])

    def reset_all_seeds(self, seed=42):
        # PyTorch CPU随机数生成器
        torch.manual_seed(seed)

        # PyTorch GPU随机数生成器
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def update_target_net(self):
        pass


    def learn(self):
        mode = 'Train'
        batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])

        prob = np.random.random()
        if prob >= 1- self.Mix_ratio:
            self.updateA, self.updateB, self.updateAB = False, False, True

        elif prob >= self.updateAProbability:
            self.updateA, self.updateB, self.updateAB = True, False, False

        else:
            self.updateA, self.updateB, self.updateAB = False, True, False


        q_A, q_B = self.compute_q(batch)


        q_target_A, q_target_B = self.compute_q_target(batch)


        if q_A is not None:
            # Compute loss
            lossA = self.loss(q_A, q_target_A)
            # Take an optimization step

            self.optimizer[self.A].zero_grad()
            lossA.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[self.A].parameters(), self.gradient_clip)

            self.optimizer[self.A].step()
            # if self.show_tb:
            #     self.logger.add_scalar(f'LossA', lossA.item(), self.step_count)

        if q_B is not None:
            # Compute loss
            lossB = self.loss(q_B, q_target_B)
            # Take an optimization step

            self.optimizer[self.B].zero_grad()
            lossB.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.Q_net[self.B].parameters(), self.gradient_clip)

            self.optimizer[self.B].step()

            # if self.show_tb:
            #     self.logger.add_scalar(f'LossB', lossB.item(), self.step_count)



        # if self.show_tb:
        #     if q_A is not None:
        #         self.logger.add_scalar(f'Targetmean/Across', q_target_A.mean().item(), self.step_count)
        #         self.logger.add_scalar(f'Targetvar/Across', q_target_A.var().item(), self.step_count)
        #     if q_B is not None:
        #         self.logger.add_scalar(f'Targetmean/Bcross', q_target_B.mean().item(), self.step_count)
        #         self.logger.add_scalar(f'Targetvar/Bcross', q_target_B.var().item(), self.step_count)






    def compute_q_target(self, batch):
        with torch.no_grad():
            q_target_A, q_target_B = None, None
            if self.updateA:
                best_actions = self.Q_net[self.A](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net[self.B](batch.next_state).gather(1, best_actions).squeeze()
                q_target_A = batch.reward + self.discount * q_next * batch.mask

            elif self.updateB:
                best_actions = self.Q_net[self.B](batch.next_state).argmax(1).unsqueeze(1)
                q_next = self.Q_net[self.A](batch.next_state).gather(1, best_actions).squeeze()
                q_target_B = batch.reward + self.discount * q_next * batch.mask

            elif self.updateAB:
                best_actions_A = self.Q_net[self.A](batch.next_state).argmax(1).unsqueeze(1)
                q_nextA = self.Q_net[self.A](batch.next_state).gather(1, best_actions_A).squeeze()
                q_target_A = batch.reward + self.discount * q_nextA * batch.mask

                best_actions_B = self.Q_net[self.B](batch.next_state).argmax(1).unsqueeze(1)
                q_nextB = self.Q_net[self.B](batch.next_state).gather(1, best_actions_B).squeeze()
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
        for Q_net in self.Q_net[0:4]:
            q_values = Q_net(state)
            q_values_list.append(q_values)
        q_values = torch.cat(q_values_list, dim=0)
        q_values = q_values.mean(dim=0, keepdim=True)
        q_values = to_numpy(q_values).flatten()

        return q_values

    def quick_parameter_diff_check(self, model_A, model_B):
        with torch.no_grad():
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

                if not torch.equal(param_A, param_B):
                    max_diff = torch.max(torch.abs(param_A - param_B)).item()
                    different_params.append((name_A, f"数值差异 (最大: {max_diff:.4f})"))

            if not different_params:
                print("✅ 所有参数相同")
            else:
                print(f"❌ 发现 {len(different_params)} 个参数不同:")
                for param_name, reason in different_params:
                    print(f"   - {param_name}: {reason}")



    def save_model_parameters(self, model, filepath):
        """
        保存模型参数到文件

        Args:
            model: PyTorch模型
            filepath: 参数保存路径
        """


        # 保存模型参数
        torch.save(model.state_dict(), filepath)
        print(f"模型参数已保存到: {filepath}")

    def load_model_parameters(self, model, filepath):
        """
        从文件加载模型参数

        Args:
            model: PyTorch模型
            filepath: 参数文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参数文件不存在: {filepath}")

        model.load_state_dict(torch.load(filepath))
        print(f"模型参数已从 {filepath} 加载")

