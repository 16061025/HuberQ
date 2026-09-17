import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt;
from torch.onnx.symbolic_opset11 import unsqueeze

from agents import Target2DDQN
from utils.sweeper import Sweeper

#plt.style.use('seaborn-ticks')
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Set font family, bold, and font size
# font = {'family':'normal', 'weight':'normal', 'size': 12}
font = {'size': 15}
matplotlib.rc('font', **font)
# Avoid Type 3 fonts in matplotlib plots: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils.helper import make_dir
from utils.plotter import read_file, get_total_combination, symmetric_ema
import os

#table_latex_file_path = os.path.join(".\\figures", "latex_tab.txt")

class ExperimentData:
    """实验数据类"""

    def __init__(self, env_name, data_dict):
        self.env_name = env_name
        self.data_dict = data_dict  # {method_name: {'x_mean': [...], 'y_mean': [...]}}

class mycolorclass:
    def __init__(self):
        self.black = '#000000'
        self.blue = '#090EE7'
        self.red = '#ff0000'
        self.green = '#66ff66'
        self.yellow = '#ffcc00'
        self.pink = '#ff00ff'
        self.brown = '#990000'
        self.molvse = '#385723'
        self.qianlanse = '#39b6c7'
        self.purple = '#9900ff'
        self.lianghuangse = '#EEF30D'




def plot_baseline_figure_13(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    baseline_methods : list
        基线方法名称列表
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行2列
    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.05,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.17,  # 底边距 (默认0.11)
        top=0.8,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.38  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案
    mycolor = mycolorclass()
    color_dict = {
        "OrderDQN": mycolor.brown,
        "DQN": mycolor.blue,
        "DDQN": mycolor.green,
        "AveragedDQN": mycolor.pink,
        "MaxminDQN": mycolor.molvse,
        "WeightedDQN": mycolor.qianlanse,
        "EBDQN": mycolor.purple,
    }

    method2lable = {
        "OrderDQN": 'Order DQN',
        "DQN": 'DQN',
        "DDQN": 'DDQN',
        "AveragedDQN": 'Averaged DQN',
        "MaxminDQN": 'Maxmin DQN',
        "WeightedDQN": 'Weighted DDQN',
        "EBDQN": 'EBDQN',

    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 3
        col = env_idx % 3
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2000, -1000)
        if env_name == "SpaceInvaders":
            ax.set_ylim(20, 45)
        if env_name == "Pixelcopter":
            ax.set_ylim(10, 30)
        if env_name == "Seaquest":
            #ax.set_ylim(-100, -100)
            pass
        # if env_name == "Breakout":
        #     ax.set_ylim(10, 12.5)

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(20)  # 调整指数字体大小
        #ax.yaxis.get_offset_text().set_weight('bold')  # 设置指数加粗
        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 找到基线+我们的方法中最好的方法
        # 首先找到我们的方法中最好的（基于最后一个y_mean值）
        best_our_method = None
        best_our_value = -float('inf')

        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                original_method_y_mean_len = len(y_mean)
                start_index = int(original_method_y_mean_len * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()

                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # 绘制基线方法 + 最好的我们的方法
        # 绘制基线方法
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method

            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        # 绘制最好的我们的方法
        x_mean = data.data_dict[best_our_method]['x_mean']
        y_mean = data.data_dict[best_our_method]['y_mean']
        y_ci = data.data_dict[best_our_method]['y_ci']
        # 只在第一列的子图添加图例标签
        label = f"DMDDQN(Our Best)"
        ax.plot(x_mean, y_mean,
                color='red',  # 最好的方法用红色突出
                linewidth=1.5,
                label=label)  # 确保在最上层
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor='red', alpha=0.5)

        if env_idx in [0]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('A')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)
        #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    legend_handles = []
    legend_labels = []
    for method in baseline_methods:
        label = method2lable[method]
        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)
    label = f"DMDDQN(Our Best)"
    line, = plt.plot([], [], color=mycolor.red,
                     linewidth=3, label=label)
    legend_handles.append(line)
    legend_labels.append(label)



    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=1.8,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),  # 在图形正上方
               columnspacing=4.8,
               ncol=4,
               fontsize=17,
               frameon=True)


    fig.savefig(f'./figures/deepbaseline1_3.png', dpi=300)

    return fig, axes

def plot_baseline_figure_14(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    baseline_methods : list
        基线方法名称列表
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行2列
    fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.056,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.2,  # 底边距 (默认0.11)
        top=0.79,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.38  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案
    mycolor = mycolorclass()
    color_dict = {
        "OrderDQN": mycolor.brown,
        "DQN": mycolor.blue,
        "DDQN": mycolor.green,
        "AveragedDQN": mycolor.pink,
        "MaxminDQN": mycolor.molvse,
        "WeightedDQN": mycolor.qianlanse,
        "EBDQN": mycolor.purple,
        "ECDQN10": mycolor.yellow,
        "ECDQN50": mycolor.green,
        "ECDQN100": mycolor.molvse,
        "ECDQN200": mycolor.red,
    }

    method2lable = {
        "OrderDQN": 'Order DQN',
        "DQN": 'DQN',
        "DDQN": 'DDQN',
        "AveragedDQN": 'Averaged DQN',
        "MaxminDQN": 'Maxmin DQN',
        "WeightedDQN": 'Weighted DDQN',
        "EBDQN": 'EBDQN',

    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 4
        col = env_idx % 4
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2000, -1000)
        if env_name == "SpaceInvaders":
            ax.set_ylim(20, 45)
        if env_name == "Pixelcopter":
            ax.set_ylim(10, 30)
        if env_name == "Seaquest":
            #ax.set_ylim(-100, -100)
            pass
        # if env_name == "Breakout":
        #     ax.set_ylim(10, 12.5)

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(10)  # 调整指数字体大小
        #ax.yaxis.get_offset_text().set_weight('bold')  # 设置指数加粗
        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 找到基线+我们的方法中最好的方法
        # 首先找到我们的方法中最好的（基于最后一个y_mean值）
        best_our_method = None
        best_our_value = -float('inf')

        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                original_method_y_mean_len = len(y_mean)
                start_index = int(original_method_y_mean_len * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()

                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # 绘制基线方法 + 最好的我们的方法
        # 绘制基线方法
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method

            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        # 绘制最好的我们的方法
        x_mean = data.data_dict[best_our_method]['x_mean']
        y_mean = data.data_dict[best_our_method]['y_mean']
        y_ci = data.data_dict[best_our_method]['y_ci']
        # 只在第一列的子图添加图例标签
        label = f"Huber DQN(Our Best)"
        ax.plot(x_mean, y_mean,
                color='red',  # 最好的方法用红色突出
                linewidth=1.5,
                label=label)  # 确保在最上层
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor='red', alpha=0.5)

        if env_idx in [0]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('A')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)
        # ax.set_xlabel(f'Steps ($t \; x1e5$)', fontsize=13)
        # ax.text(0.5, -0.125, f"\n({figcharidx}) {env_name}",
        #          transform=ax.transAxes,  # 用轴坐标（0~1）
        #          ha='center', va='top',  # 水平居中，顶部对齐
        #          fontsize=20)
    legend_handles = []
    legend_labels = []
    for method in baseline_methods:
        label = method2lable[method]
        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)
    label = f"Huber DQN(Our Best)"
    line, = plt.plot([], [], color=mycolor.red,
                     linewidth=3, label=label)
    legend_handles.append(line)
    legend_labels.append(label)



    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=1.8,
               handletextpad=0.4,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),  # 在图形正上方
               columnspacing=4.9,
               ncol=4,
               fontsize=17,
               frameon=True)


    ax = axes[0, 3]
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(10)  # 调整指数字体大小
    ax.set_xlim((0, 1000000))
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                  ['0', '2', '4', '6', '8', '10'])
    ax.grid()
    para_data = None
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        if env_name == "Breakout":
            para_data = data
            break
    # 绘制我们的方法

    for method in our_methods:
        if method[5:] not in ["10", "50", "100", "200"]:
            continue
        x_mean = para_data.data_dict[method]['x_mean']
        y_mean = para_data.data_dict[method]['y_mean']
        y_ci = para_data.data_dict[method]['y_ci']

        # 只在第一列的子图添加图例标签
        label = method
        ax.plot(x_mean, y_mean,
                color=color_dict[method],
                linewidth=1.5,
                linestyle='-',  # 基线方法用虚线
                label=label)
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)
    ax.set_xlabel(f'Steps ($t \; x1e5$)\n(D) Breakout', fontsize=20)
    # ax.set_xlabel(f'Steps ($t \; x1e5$)', fontsize=13)
    # ax.text(0.5, -0.125, f"\n(D) Breakout",
    #         transform=ax.transAxes,  # 用轴坐标（0~1）
    #         ha='center', va='top',  # 水平居中，顶部对齐
    #         fontsize=20)

    legend_handles = []
    legend_labels = []
    for method in our_methods:
        label = f"$\delta$={method[5:]}"
        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        ncol=2,
        fontsize=11.5,
    )

    fig.savefig(f'./figures/deepbaseline1_4.png', dpi=300)

    return fig, axes

def plot_para_figure_13(data_list, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行2列
    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.05,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.2,  # 底边距 (默认0.11)
        top=0.83,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.38  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案
    mycolor = mycolorclass()
    color_dict = {
        "ECDQN10":  mycolor.yellow ,
        "ECDQN50": mycolor.green ,
        "ECDQN100": mycolor.molvse ,
        "ECDQN200": mycolor.blue ,
        # "ECDQN0.4": mycolor.red ,
        # "ECDQN0.5": mycolor.lianghuangse ,
        # "ECDQN0.6": mycolor.qianlanse ,
        # "ECDQN0.7": mycolor.purple,
        # "ECDQN0.8": mycolor.pink ,
        # "ECDQN0.9": mycolor.yellow ,
        # "ECDQN1":   mycolor.brown ,
    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 3
        col = env_idx % 3
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2500, -1000)
        if env_name == "SpaceInvaders":
            ax.set_ylim(15, 45)
        if env_name == "Pixelcopter":
            ax.set_ylim(10, 30)
        if env_name == "Seaquest":
            ax.set_ylim(0, 1.2)

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(20)  # 调整指数字体大小
        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 绘制我们的方法

        for method in our_methods:
            if method[5:] not in ["10","50", "100", "200"]:
                continue
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method
            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)


        if env_idx in [0]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('A')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)
        #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    legend_handles = []
    legend_labels = []
    for method in our_methods:
        # if method == 'M-DDQN0' or method == 'M-DDQN1':
        #     rho = float(method[-1])
        # else:
        #     rho = float(method[-3:])
        delta = int(method[5:])
        label = r"$\delta=$"+f'{delta:d}'

        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)

    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=2.5,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),  # 在图形正上方
               columnspacing=8,
               ncol=6,
               fontsize=17,
               frameon=True)

    fig.savefig(f'./figures/deeppara1_3.png', dpi=300)

    return fig, axes


def plot_baseline_figure(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    baseline_methods : list
        基线方法名称列表
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，1行4列
    fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.04,  # 左边距 (默认0.125)
        right=0.99,  # 右边距
        bottom=0.19,  # 底边距 (默认0.11)
        top=0.87,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.25  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案
    mycolor = mycolorclass()
    color_dict = {
        "OrderDQN": mycolor.brown,
        "DQN": mycolor.blue,
        "DDQN": mycolor.green,
        "AveragedDQN": mycolor.pink,
        "MaxminDQN": mycolor.molvse,
        "WeightedDQN": mycolor.qianlanse,
        "EBDQN": mycolor.purple,
    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 4
        col = env_idx % 4
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2000, -1000)
        if env_name == "SpaceInvaders":
            ax.set_ylim(20, 45)
        if env_name == "Pixelcopter":
            ax.set_ylim(10, 30)

        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 找到基线+我们的方法中最好的方法
        # 首先找到我们的方法中最好的（基于最后一个y_mean值）
        best_our_method = None
        best_our_value = -float('inf')

        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                original_method_y_mean_len = len(y_mean)
                start_index = int(original_method_y_mean_len * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()

                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # 绘制基线方法 + 最好的我们的方法
        # 绘制基线方法
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method
            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        # 绘制最好的我们的方法
        x_mean = data.data_dict[best_our_method]['x_mean']
        y_mean = data.data_dict[best_our_method]['y_mean']
        y_ci = data.data_dict[best_our_method]['y_ci']
        # 只在第一列的子图添加图例标签
        label = f"OurBest"
        ax.plot(x_mean, y_mean,
                color='red',  # 最好的方法用红色突出
                linewidth=1.5,
                label=label)  # 确保在最上层
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor='red', alpha=0.5)

        if env_idx in [0]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('a')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps(x1e5)\n({figcharidx}) Comparsion on {env_name}', fontsize=20)
        #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    legend_handles = []
    legend_labels = []
    for method in baseline_methods:
        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=method)
        legend_handles.append(line)
        legend_labels.append(method)
    line, = plt.plot([], [], color=mycolor.red,
                     linewidth=3, label="ourbest")
    legend_handles.append(line)
    legend_labels.append("ourbest")



    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=2.3,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),  # 在图形正上方
               columnspacing=3.5,
               ncol=8,
               fontsize=17,
               frameon=True)


    fig.savefig(f'./figures/deepbaseline1_4.png', dpi=300)

    return fig, axes

def plot_para_figure(data_list, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行N列
    fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.04,  # 左边距 (默认0.125)
        right=0.99,  # 右边距
        bottom=0.19,  # 底边距 (默认0.11)
        top=0.87,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.25  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案
    mycolor = mycolorclass()
    color_dict = {
        "M-DDQN0":   mycolor.black ,
        "M-DDQN0.1": mycolor.green ,
        "M-DDQN0.2": mycolor.molvse ,
        "M-DDQN0.3": mycolor.blue ,
        "M-DDQN0.4": mycolor.red ,
        "M-DDQN0.5": mycolor.pink ,
        "M-DDQN0.6": mycolor.qianlanse ,
        "M-DDQN0.7": mycolor.purple,
        "M-DDQN0.8": mycolor.lianghuangse ,
        "M-DDQN0.9": mycolor.yellow ,
        "M-DDQN1":   mycolor.brown ,
    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 4
        col = env_idx % 4
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2500, -1000)
        if env_name == "SpaceInvaders":
            ax.set_ylim(15, 45)
        if env_name == "Pixelcopter":
            ax.set_ylim(10, 30)
        if env_name == "Seaquest":
            ax.set_ylim(0, 1.2)

        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 绘制我们的方法

        for method in our_methods:
            if method[-3:] not in ["0.1","0.2", "0.3", "0.4", "0.5","0.6", "0.7", "0.8", "0.9", "QN0", "QN1"]:
                continue
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method
            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)


        if env_idx in [0]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('a')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps(x1e5)\n({figcharidx}) Comparsion on {env_name}', fontsize=20)
        #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    legend_handles = []
    legend_labels = []
    for method in our_methods:
        if method == 'M-DDQN0' or method == 'M-DDQN0.1' or method == 'M-DDQN1':
            rho = float(method[-1])
        else:
            rho = float(method[-3:])
        label = r"$\rho=$"+f'{rho:.1f}'

        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)

    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=1.5,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),  # 在图形正上方
               columnspacing=2.5,
               ncol=11,
               fontsize=17,
               frameon=True)

    fig.savefig(f'./figures/deeppara1_4.png', dpi=300)

    return fig, axes

def plot_paper_figure(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    baseline_methods : list
        基线方法名称列表
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行N列
    fig, axes = plt.subplots(2, 4, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.06,  # 左边距 (默认0.125)
        right=0.99,  # 右边距
        bottom=0.11,  # 底边距 (默认0.11)
        top=0.99,  # 顶边距
        wspace=0.3,  # 水平间距 (默认0.2)
        hspace=0.35  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案

    color_dict = {
        "OrderDQN": 'c',
        "DQN": 'm',
        "DDQN": 'y',
        "AveragedDQN": '#1f77b4',
        "EnsembleDQN": '#ff7f0e',
        "MaxminDQN": 'purple',
        "WeightedDQN": 'olive',
        "EBDQN": 'lime',
        "M-DDQN0": 'orange',
        "M-DDQN0.2": 'brown',
        "M-DDQN0.4": 'red',
        "M-DDQN0.6": 'blue',
        "M-DDQN0.8": 'green',
        "M-DDQN1": 'pink',

    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 4
        col = env_idx % 4
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-400, -50)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2500, -1000)

        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 找到基线+我们的方法中最好的方法
        # 首先找到我们的方法中最好的（基于最后一个y_mean值）
        best_our_method = None
        best_our_value = -float('inf')

        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                original_method_y_mean_len = len(y_mean)
                start_index = int(original_method_y_mean_len * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()

                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # 绘制基线方法 + 最好的我们的方法
        # 绘制基线方法
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method
            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        # 绘制最好的我们的方法
        x_mean = data.data_dict[best_our_method]['x_mean']
        y_mean = data.data_dict[best_our_method]['y_mean']
        y_ci = data.data_dict[best_our_method]['y_ci']
        # 只在第一列的子图添加图例标签
        label = f"OurBest"
        ax.plot(x_mean, y_mean,
                color='red',  # 最好的方法用红色突出
                linewidth=1.5,
                label=label)  # 确保在最上层
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        if env_idx in [0, 4]:
            ax.set_ylabel('Average Return', fontsize=15)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('a')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps(x1e5)\n({figcharidx}) Comparsion on {env_name}', fontsize=15)
        ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    #最后两个图
    rho_envs = ["Asterix", "SpaceInvaders"]
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        if env_name not in rho_envs:
            continue
        subfigidx = rho_envs.index(env_name) + 6
        row = subfigidx // 4
        col = subfigidx % 4
        ax = axes[row, col]


        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                      ['0', '2', '4', '6', '8', '10'])
        #ax.grid()


        # 所有我们的方法
        for method in our_methods:
            if method[-3:] not in ["0.2", "0.4", "0.6", "0.8", "QN0", "QN1"]:
                continue
            if method in data.data_dict:
                x_mean = data.data_dict[method]['x_mean']
                y_mean = data.data_dict[method]['y_mean']
                y_ci = data.data_dict[method]['y_ci']
                # 设置线宽：最好的方法加粗
                linewidth = 2.5 if method == best_our_method else 1.5

                label = method
                ax.plot(x_mean, y_mean,
                        color=color_dict[method],
                        linewidth=linewidth,
                        label=label)
                ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)



        # 设置轴标签
        figcharidx = chr(ord('a') + subfigidx)
        # 设置轴标签
        ax.set_xlabel(f'Steps(x1e5)\n({figcharidx}) Effect of '+ r"""$\rho$"""+f' on {env_name}', fontsize=15)
        # 添加网格
        ax.grid()
        ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

        # # 自动调整y轴范围，留出一些边距


    # 创建图例
    # 第一行图例（基线方法 + 最好的我们的方法）
    # top_legend_handles = []
    # top_legend_labels = []
    #
    # # 基线方法
    # for method in baseline_methods:
    #     line, = plt.plot([], [], color=color_dict[method],
    #                      linewidth=1.5, linestyle='-', label=method)
    #     top_legend_handles.append(line)
    #     top_legend_labels.append(method)
    #
    # # 最好的我们的方法
    # line_best, = plt.plot([], [], color='red',
    #                       linewidth=2.5, label='OurBest')
    # top_legend_handles.append(line_best)
    # top_legend_labels.append('OurBest')
    #
    # showed_our_methods = [f'M-DDQN{i:.1f}' for i in np.arange(0.2, 1, 0.2)]
    # showed_our_methods += ['M-DDQN0', 'M-DDQN1']
    #
    # for method in showed_our_methods:
    #     line, = plt.plot([], [], color=color_dict[method],
    #                      linewidth=1.5, label=method)
    #     top_legend_handles.append(line)
    #     top_legend_labels.append(method)
    #
    # # 添加第一行图例（居中，在图形上方）
    # fig.legend(handles=top_legend_handles,
    #            labels=top_legend_labels,
    #            loc='upper center',
    #            bbox_to_anchor=(0.5, 0.99),  # 在图形正上方
    #            ncol=len(top_legend_handles) // 2 if len(top_legend_handles) % 2 == 0 else len(
    #                top_legend_handles) // 2 + 1,
    #            fontsize=15,
    #            frameon=True)

    # 调整布局，为图例留出空间
    #plt.tight_layout(rect=[0, 0, 1, 1])  # 顶部留出10%空间给图例

    return fig, axes

class Plotter(object):
    def __init__(self, cfg):
        cfg.setdefault('ci', None)
        self.x_label = cfg['x_label']
        self.y_label = cfg['y_label']
        self.show = cfg['show']
        self.imgType = cfg['imgType']
        self.ci = cfg['ci']
        self.runs = cfg['runs']
        make_dir('figures/')

    def get_result(self, exp, config_idx, mode):
        '''
        Given exp and config index, get the results
        '''
        total_combination = get_total_combination(exp)
        result_list = []
        for _ in range(self.runs):
            result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
            # If result file exist, read and merge
            result = read_file(result_file)
            if result is not None:
                # Add config index as a column
                result['Config Index'] = config_idx
                result_list.append(result)
            config_idx += total_combination

        # Do symmetric EMA (exponential moving average)
        # Get x's and y's in form of numpy arries
        xs, ys = [], []
        for result in result_list:
            xs.append(result[self.x_label].to_numpy())
            ys.append(result[self.y_label].to_numpy())
        # Do symetric EMA to get new x's and y's
        low = max(x[0] for x in xs)
        high = min(x[-1] for x in xs)
        n = min(len(x) for x in xs)
        for i in range(len(xs)):
            new_x, new_y, _ = symmetric_ema(xs[i], ys[i], low, high, n)
            result_list[i] = result_list[i][:n]
            result_list[i].loc[:, self.x_label] = new_x
            result_list[i].loc[:, self.y_label] = new_y

        ys = []
        for result in result_list:
            ys.append(result[self.y_label].to_numpy())
        # Compute x_mean, y_mean and y_ci
        ys = np.array(ys)
        x_mean = result_list[0][self.x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        if self.ci == 'sd':
            y_ci = np.std(ys, axis=0, ddof=0)
        elif self.ci == 'se':
            y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))

        return x_mean, y_mean, y_ci


def x_format(x, pos):
    # return '$%.1f$x$10^{6}$' % (x/1e6)
    return '%.1f' % (x / 1e6)


cfg = {
    'x_label': 'Step',
    'y_label': 'Average Return',
    'show': False,
    'imgType': 'png',
    'ci': 'se',
    'x_format': None,
    'y_format': None,
    'xlim': {'min': None, 'max': None},
    'ylim': {'min': None, 'max': None},
    'runs': 10,
    'loc': 'lower right'
}


def get_num_combinations_of_dict_before_key(config_dict, target_key_seq):
    '''
    Get # of combinations for configurations in a config dict
    '''
    assert type(config_dict) == dict, 'Config file must be a dict!'
    num_combinations_of_dict = 1
    num_combinations_of_key = None
    for key, values in config_dict.items():
        if key == target_key_seq[0]:
            if len(target_key_seq) == 1:
                num_combinations_of_key = len(values)
                break
            else:
                num_combinations_of_list, num_combinations_of_key = get_num_combinations_of_list(values, target_key_seq[1:])
                num_combinations_of_dict *= num_combinations_of_list
                if num_combinations_of_key is not None:
                    break


        else:
            num_combinations_of_list, num_combinations_of_key = get_num_combinations_of_list(values, target_key_seq)
            num_combinations_of_dict *= num_combinations_of_list
            if num_combinations_of_key is not None:
                break

    config_dict['num_combinations'] = num_combinations_of_dict
    return num_combinations_of_dict, num_combinations_of_key


def get_num_combinations_of_list(config_list, target_key):
    '''
    Get # of combinations for configurations in a config list
    '''
    assert type(config_list) == list, 'Elements in a config dict must be a list!'
    num_combinations_of_list = 0
    num_combinations_of_key = None
    for value in config_list:
        if type(value) == dict:
            if not ('num_combinations' in value.keys()):
                _, num_combinations_of_key = get_num_combinations_of_dict_before_key(value, target_key)
            num_combinations_of_list += value['num_combinations']
        else:
            num_combinations_of_list += 1
    return num_combinations_of_list, num_combinations_of_key

def cluster_single_exp_idx(exp):
    config_file_path = f'./configs/{exp}.json'
    config = json.load(open(config_file_path, 'r'))
    total_combination = get_total_combination(exp)
    if "_dqn" in exp or "_Vanilladqn" in exp:
        key_seq = ["agent", "name"]
    elif "rho" in exp:
        key_seq = ["agent", "M"]
    elif "Maxmin" in exp or "Averaged" in exp or "Ensemble" in exp:
        key_seq = ["agent", "target_networks_num"]
    elif "Weighted" in exp:
        key_seq = ["agent", "c"]
    elif "MixData" in exp or "Target2MixDataDDQN" in exp:
        key_seq = ["agent", "rho"]
    elif "Target2DDQN" in exp:
        key_seq = ["agent", "sel"]
    elif "AdaODQN" in exp:
        key_seq = ["agent", "m"]
    elif "EBDQN" in exp:
        key_seq = ["agent", "target_networks_num"]
    elif "ErrorClipDQN" in exp:
        key_seq = ["agent", "delta"]
    else:
        raise NotImplementedError

    num_combinations_before_key, num_combinations_of_key = get_num_combinations_of_dict_before_key(config, key_seq)
    num_combinations_after_key = int(total_combination / (num_combinations_before_key * num_combinations_of_key))
    all_keys_group = []
    for i in range(num_combinations_of_key):
        key_group = []
        row_start_idx = i * num_combinations_after_key

        for j in range(num_combinations_before_key):
            start_idx = row_start_idx + (j*num_combinations_after_key*num_combinations_of_key)
            end_idx = start_idx + num_combinations_after_key
            idxs = range(start_idx, end_idx)
            key_group+=idxs
        all_keys_group.append(key_group)

    all_keys_group = np.array(all_keys_group).T+1 #make idx begin from 1 not 0
    ##N_setting every setting has idx

    result = np.vectorize(lambda x: (exp, x), otypes=[object])(all_keys_group)

    return result

def cluster_exp_idx(exp_list):
    multi_exp_idx_group_list = [] #
    for exp in exp_list:
        exp_idx_group_list = cluster_single_exp_idx(exp) # np: one element correspond to one type env setting
        multi_exp_idx_group_list.append(exp_idx_group_list)
    if len(multi_exp_idx_group_list) == 0:
        return []
    exp_idx_group_list = np.concatenate(multi_exp_idx_group_list, axis=1)
    return exp_idx_group_list




def generate_figure_by_method_metric(methods, metrics, title="env"):


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        'figure.dpi': 200,
    })

    color_list = plt.cm.tab20.colors[:len(methods)]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    all_method_x_mean = metrics["x_mean"]
    all_method_y_mean = metrics["y_mean"]
    all_method_y_ci = metrics["y_ci"]
    for i in range(len(methods)):
        label = methods[i]
        x_mean, y_mean, y_ci = all_method_x_mean[i], all_method_y_mean[i], all_method_y_ci[i]
        first_index_search = np.searchsorted(x_mean, 5e5, side='right')
        # x_mean = x_mean[0:first_index_search]
        # y_mean = y_mean[0:first_index_search]
        # y_ci = y_ci[0:first_index_search]
        plt.plot(x_mean, y_mean, linewidth=1.5, color=color_list[i], label=label)
        if cfg['ci'] in ['se', 'sd']:
            plt.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_list[i], alpha=0.5)
            # Set x and y axis
    # ax.set_xlabel("Steps (x$10^{6}$)", fontsize=16)
    # ax.set_ylabel('Average Return', fontsize=16, rotation='horizontal')
    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel('Average Return', fontsize=16)
    ax.set_ylim(-1250, -1050)
    plt.yticks(size=11)
    plt.xticks(size=11)
    plt.title(title, fontsize=16)
    # # Set legend
    ax.legend(loc=cfg['loc'], frameon=False, fontsize=12)
    # Adjust layout automatically
    plt.tight_layout()
    # Save and show

    image_path = f'./figures/{title}.{cfg["imgType"]}'
    ax.get_figure().savefig(image_path)

    plt.show()

def create_latex_table(row_names, data_dict, title="table", label="tab:my_table"):
    """
    生成LaTeX三线表代码
    row_names: method names
    data_dict: metric_name:methods_metric_value
    """
    col_names = list(data_dict.keys())

    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * len(col_names) + "}")
    latex_code.append(r"\hline ")

    # 表头 - 左上角空白
    header = " & ".join([""] + col_names) + r" \\"
    latex_code.append(header)
    latex_code.append(r"\hline ")

    # 数据行
    for i, row_name in enumerate(row_names):
        row_data = [row_name]
        for col_name in col_names:
            data = data_dict[col_name]
            sorted_idx = np.argsort(data)[::-1]
            if col_name == "improve":
                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]*100:.2f}}}" + r'\%')  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]*100:.2f}}}" + r'\%')  # 次大值下划线
                else:
                    row_data.append(f"{data[i]*100:.2f}" + r'\%')
            else:

                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次大值下划线
                else:
                    row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)

def table_experiment_comparison(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    生成LaTeX三线表代码
    row_names: method names
    data_dict: metric_name:methods_metric_value
    """


    env_names = []
    for env in data_list:
        env_names.append(env.env_name)

    row_names = list(data_list[0].data_dict.keys())

    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * len(env_names)*2 + "}")
    latex_code.append(r"\hline ")

    # 表头 - 左上角空白

    header = " & ".join([""]+[r"\multicolumn{2}{c}{"+env_name+"}" for env_name in env_names]) + r" \\"
    latex_code.append(header)

    hline = r""
    for i in range(len(data_list)):
        hline += r" \cmidrule(lr){"+str(i*2+2)+"-"+str(i*2+3)+"}"
    latex_code.append(hline)


    col_names = []
    for env in data_list:
        col_names.append("Reward")
        col_names.append("Improv")

    title = " & ".join(["Algorithm"]+col_names) + r" \\"
    latex_code.append(title)

    latex_code.append(r"\hline ")


    table_data = np.zeros((len(row_names), len(env_names)*2))
    for j, env in enumerate(data_list):
        for i, method in enumerate(row_names):
            env_method_y_mean = env.data_dict[method]["y_mean"]
            original_method_y_mean_len = len(env_method_y_mean)
            start_index = int(original_method_y_mean_len * 0.9)
            table_method_y_mean = np.array(env_method_y_mean[start_index:]).mean()
            table_data[i][j*2] = table_method_y_mean

    for j, env in enumerate(data_list):
        reward_col = table_data[:, j * 2]

        best_baseline_value = float("-inf")
        #找出最大的baseline的值
        for i, method in enumerate(row_names):
            if method in our_methods:
                continue
            if reward_col[i] > best_baseline_value:
                best_baseline_value = reward_col[i]

        ref_baseline_value = reward_col[0]
        ref_baseline_value = best_baseline_value
        improve_col = (reward_col - ref_baseline_value)/np.abs(ref_baseline_value)
        table_data[:,j*2+1] = improve_col
    # 数据行
    for i, row_name in enumerate(row_names):
        if row_name in our_methods:
            if row_name in ["M-DDQN1", "M-DDQN0"]:
                row_name = r"$\rho="+str(row_name[-1])+r"$"
            else:
                row_name = r"$\rho="+str(row_name[-3:])+r"$"
        row_data = [row_name]


        for j, col_name in enumerate(col_names):
            data = table_data[:,j]
            sorted_idx = np.argsort(data)[::-1]
            if col_name == "Improv":
                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]*100:.2f}}}" + r'\%')  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]*100:.2f}}}" + r'\%')  # 次大值下划线
                else:
                    row_data.append(f"{data[i]*100:.2f}" + r'\%')
            else:

                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次大值下划线
                else:
                    row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)

def table_experiment_comparison_only_reward(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    生成LaTeX三线表代码
    row_names: method names
    data_dict: metric_name:methods_metric_value
    """


    env_names = []
    for env in data_list:
        env_names.append(env.env_name)

    row_names = list(data_list[0].data_dict.keys())

    row_names = ['AveragedDQN', 'WeightedDQN', "DDQN", "DQN", "OrderDQN", "EBDQN", "MaxminDQN"] + \
                 ["ECDQN10","ECDQN50","ECDQN100","ECDQN200",]



    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * len(env_names)*1 + "}")
    latex_code.append(r"\hline ")

    # 表头 - 左上角空白

    col_names = []
    for env in data_list:
        col_names.append("Reward")
        #col_names.append("Improv")

    title = " & ".join(["Algorithm"]+env_names) + r" \\"
    latex_code.append(title)

    latex_code.append(r"\hline ")


    table_data = np.zeros((len(row_names), len(env_names)*1))
    for j, env in enumerate(data_list):
        for i, method in enumerate(row_names):
            env_method_y_mean = env.data_dict[method]["y_mean"]
            original_method_y_mean_len = len(env_method_y_mean)
            start_index = int(original_method_y_mean_len * 0.9)
            table_method_y_mean = np.array(env_method_y_mean[start_index:]).mean()
            table_data[i][j*1] = table_method_y_mean

    # for j, env in enumerate(data_list):
    #     reward_col = table_data[:, j * 2]
    #
    #     best_baseline_value = float("-inf")
    #     #找出最大的baseline的值
    #     for i, method in enumerate(row_names):
    #         if method in our_methods:
    #             continue
    #         if reward_col[i] > best_baseline_value:
    #             best_baseline_value = reward_col[i]
    #
    #     ref_baseline_value = reward_col[0]
    #     ref_baseline_value = best_baseline_value
    #     improve_col = (reward_col - ref_baseline_value)/np.abs(ref_baseline_value)
    #     table_data[:,j*2+1] = improve_col
    # 数据行
    for i, row_name in enumerate(row_names):

        row_data = [row_name]


        for j, col_name in enumerate(col_names):
            data = table_data[:,j]
            sorted_idx = np.argsort(data)[::-1]
            if col_name == "Improv":
                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]*100:.2f}}}" + r'\%')  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]*100:.2f}}}" + r'\%')  # 次大值下划线
                else:
                    row_data.append(f"{data[i]*100:.2f}" + r'\%')
            else:

                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次大值下划线
                else:
                    row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)


def render_latex_table(table_tex_code, title):

    # 创建完整的LaTeX文档
    print(table_tex_code)
    full_latex_code = table_tex_code.replace("\n", " ")
    #print(full_latex_code)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')

    # 渲染LaTeX表格
    ax.text(0.5, 0.5, full_latex_code, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            horizontalalignment='center')

    image_path = f'./figures/{title}_table.{cfg["imgType"]}'
    ax.get_figure().savefig(image_path)
    #plt.tight_layout()
    plt.show()


def reorder_methods(method_names):
    # 分离三种类型的方法
    ddqn_mask = [m == 'DDQN' for m in method_names]
    dqn_mask = [m == 'DQN' for m in method_names]
    other_mask = [m not in ['DQN', 'DDQN'] for m in method_names]

    # 重新排序
    new_order = np.where(ddqn_mask)[0].tolist() + \
                np.where(other_mask)[0].tolist() + \
                np.where(dqn_mask)[0].tolist()
    return new_order


def generate_table_by_method_metric(methods, metrics, title):
    row_names = methods
    Reward_col = []
    all_method_y_mean = metrics["y_mean"]
    for method_y_mean in all_method_y_mean:
        original_method_y_mean_len = len(method_y_mean)
        start_index = int(original_method_y_mean_len*0.9)
        table_method_y_mean = np.array(method_y_mean[start_index:]).mean()
        Reward_col.append(table_method_y_mean)
    Reward_col = np.array(Reward_col).flatten()


    improve_col = (Reward_col - Reward_col[0]) / np.abs(Reward_col[0])

    data_dict = {"Reward": Reward_col,
                 "improve": improve_col}

    latex_code = create_latex_table(row_names, data_dict)
    # test_render_latex_table()
    render_latex_table(latex_code, title)



def get_method_metrics_by_exp_idx_gropu(expidxs_list):
    plotter = Plotter(cfg)

    template_exp, template_config_idx = expidxs_list[0]
    template_config_path = f'./logs/{template_exp}/{template_config_idx}/config.json'
    template_config = json.load(open(template_config_path, "r"))
    env = template_config["env"]['name']


    methods = []
    x_mean_metrics = []
    y_mean_metrics = []
    y_ci_metrics = []
    for i in range(len(expidxs_list)):
        exp, config_idx = expidxs_list[i]
        config_path = f'./logs/{exp}/{config_idx}/config.json'
        config = json.load(open(config_path, "r"))
        agent = config['agent']

        if "_dqn" in exp or "Vanilladqn" in exp:
            label = config['agent']['name']
        elif "Vanillarho" in exp:
            K = config['agent']['networks_num']
            M = config['agent']['M']
            label = f'VrhoK{K}M{M}'
        elif "_rho" in exp:
            agent = config['agent']['name']
            K = config['agent']['networks_num']
            M = config['agent']['M']
            label = f'rhoK{K}M{M}'
        elif "Maxmin" in exp or "Averaged" in exp or "Ensemble" in exp:
            agent = config['agent']['name']
            K = config['agent']['target_networks_num']
            label = f'{agent}{K}'
            label = f'{agent}'
        elif "Weighted" in exp:
            agent = config['agent']['name']
            c = config['agent']['c']
            label = f'{agent}{c}'
            label = f'WeightedDQN'
        elif "MixData" in exp or "Target2MixDataDDQN" in exp:
            agent = config['agent']['name']
            rho = config['agent']['rho']
            label = f'{agent}{rho}'
            label = f'M-DDQN{rho}'
        elif "Target2DDQN" in exp:
            agent = config['agent']['name']
            sel = config['agent']['sel']
            label = f'{agent}{sel}sel'
        elif "AdaODQN" in exp:
            agent = config['agent']['name']
            M = config['agent']['M']
            m = config['agent']['m']
            if int(m) != 1:
                continue
            label = f'{agent}M{M}m{m}'
            label = f'OrderDQN'
        elif "EBDQN" in exp:
            agent = config['agent']['name']
            K = config['agent']['target_networks_num']
            label = f'{agent}{K}'
            label = f'EBDQN'
        elif "ErrorClipDQN" in exp:
            agent = config['agent']['name']
            delta = config['agent']['delta']
            label = f'ECDQN{delta}'
        else:
            raise NotImplementedError
        methods.append(label)
        print(f'[{exp}]: Plot Train results: {config_idx}')
        x_mean, y_mean, y_ci = plotter.get_result(exp, config_idx, 'Train')
        x_mean_metrics.append(x_mean)
        y_mean_metrics.append(y_mean)
        y_ci_metrics.append(y_ci)

    new_order = reorder_methods(methods)

    methods = [methods[i] for i in new_order]
    metric_dict = {
        "x_mean": [x_mean_metrics[i] for i in new_order],
        "y_mean": [y_mean_metrics[i] for i in new_order],
        "y_ci": [y_ci_metrics[i] for i in new_order],
    }

    return methods, metric_dict

def getexp_idxenv(exp, confgi_idx):
    config_path = f'./logs/{exp}/{confgi_idx}/config.json'
    config = json.load(open(config_path, "r"))
    env = config['env']['name']
    if "copter" in env:
        env = "Pixelcopter"
    elif "Asterix" in env:
        env = "Asterix"
    elif "Space" in env:
        env = "SpaceInvaders"
    elif "Seaquest" in env:
        env = "Seaquest"
    elif "MountainCar" in env:
        env = "MountainCar"
    elif "PuckWorld" in env:
        env = "PuckWorld"
    elif "Breakout" in env:
        env = "Breakout"
    else:
        raise NotImplementedError
    return env

def learning_curve(exp_list, runs=1):
    cfg['runs'] = runs

    exp_idx_group_list = cluster_exp_idx(exp_list) #one element correspond to one type env setting

    for exp_idx_group in exp_idx_group_list:
        exp, idx = exp_idx_group[0]
        env = getexp_idxenv(exp, idx)
        methods, metrics = get_method_metrics_by_exp_idx_gropu(exp_idx_group)

        # new_methods = []
        # new_metrics = {
        #     "x_mean": [],
        #     "y_mean": [],
        #     "y_ci": [],
        # }
        # for i in range(len(methods)):
        #     if "MixData" in methods[i]:
        #         rho = float(methods[i].split("DQN")[-1])
        #         if rho != 0.4:
        #             continue
        #
        #     # if methods[i] == "DDQN":
        #     #     continue
        #
        #     new_methods.append(methods[i])
        #     new_metrics["x_mean"].append(metrics["x_mean"][i])
        #     new_metrics["y_mean"].append(metrics["y_mean"][i])
        #     new_metrics["y_ci"].append(metrics["y_ci"][i])
        # methods = new_methods
        # metrics = new_metrics

        # generate_figure_by_method_metric(methods, metrics, title=env)
        # generate_table_by_method_metric(methods, metrics, title=env)

        data_dict = {}
        for i in range(len(methods)):
            methodname = methods[i]
            baseline_x_mean = metrics['x_mean'][i]
            baseline_y_mean = metrics['y_mean'][i]
            baseline_y_ci = metrics['y_ci'][i]
            data_dict[methodname] = {
                'x_mean': baseline_x_mean,
                'y_mean': baseline_y_mean,
                'y_ci': baseline_y_ci,
            }
        env1 = ExperimentData(env, data_dict)

        return env1

if __name__ == "__main__":

    # exp_list = [
    #
    #     [
    #         "MERL_copter_dqn",
    #         "MERL_copter_Maxmin",
    #         "MERL_copter_Averaged",
    #         "MERL_copter_Weighted",
    #         "MERL_copter_Target2MixDataDDQN",
    #         "MERL_copter_AdaODQN",
    #         "MERL_copter_EBDQN",
    #
    #     ],
    #
    #
    #
    #     [
    #         "MERL_Asterix_dqn",
    #         "MERL_Asterix_Target2MixDataDDQN",
    #         "MERL_Asterix_Maxmin",
    #         "MERL_Asterix_Averaged",
    #         "MERL_Asterix_Weighted",
    #         "MERL_Asterix_AdaODQN",
    #         "MERL_Asterix_EBDQN",
    #
    #     ],
    #
    #
    #
    #     [
    #         "MERL_SpaceInvadersv1_dqn",
    #         "MERL_SpaceInvadersv1_Target2MixDataDDQN",
    #         "MERL_SpaceInvadersv1_Maxmin",
    #         "MERL_SpaceInvadersv1_Averaged",
    #         "MERL_SpaceInvadersv1_Weighted",
    #         "MERL_SpaceInvadersv1_AdaODQN",
    #         "MERL_SpaceInvadersv1_EBDQN",
    #
    #     ],
    #
    #     [
    #         "MERL_seaquestv1_dqn",
    #         "MERL_seaquestv1_Target2MixDataDDQN",
    #         "MERL_seaquestv1_Maxmin",
    #         "MERL_seaquestv1_Averaged",
    #         "MERL_seaquestv1_Weighted",
    #         "MERL_seaquestv1_AdaODQN",
    #         "MERL_seaquestv1_EBDQN",
    #
    #
    #     ],
    #
    #     [
    #         "MERL_mc_dqn",
    #         "MERL_mc_Target2MixDataDDQN",
    #         "MERL_mc_Maxmin",
    #         "MERL_mc_Averaged",
    #         "MERL_mc_Weighted",
    #         "MERL_mc_AdaODQN",
    #         "MERL_mc_EBDQN",
    #
    #     ],
    #
    #
    #     [
    #         "MERL_PuckWorld_dqn",
    #         "MERL_PuckWorld_Target2MixDataDDQN",
    #         "MERL_PuckWorld_Maxmin",
    #         "MERL_PuckWorld_Averaged",
    #         "MERL_PuckWorld_Weighted",
    #         "MERL_PuckWorld_AdaODQN",
    #         "MERL_PuckWorld_EBDQN",
    #
    #     ],
    #
    #
    # ]

    envlist = [
        "Asterix",
        # #"SpaceInvadersv1",
        "seaquestv1",
        "breakout",
        #"PuckWorld",

        #"copter",


        #

        #'mc',


    ]
    methodslist = [
        "dqn",
        "Weighted",
        "Maxmin",
        "Averaged",
        "AdaODQN",
        "EBDQN",
        "ErrorClipDQN",
    ]

    exp_list = []
    for env in envlist:
        envcfglist = []
        for method in methodslist:
            config_file = f'MERL_{env}_{method}'
            envcfglist.append(config_file)
        exp_list.append(envcfglist)

    baseline_methods = ['AveragedDQN', 'WeightedDQN', "DDQN", "DQN", "OrderDQN", "EBDQN", "MaxminDQN"]

    baseline_methods = ["DQN",
                        "MaxminDQN",

                        "DDQN",
                        "EBDQN",

                        'AveragedDQN',
                        "OrderDQN",

                        'WeightedDQN',  ]


    our_methods = [f'M-DDQN{i:.1f}' for i in np.arange(0.1, 1, 0.1)]
    our_methods = ['M-DDQN0']+our_methods+['M-DDQN1']
    our_methods = [
        'ECDQN10',
        'ECDQN50',

        'ECDQN100',
        'ECDQN200',
    ]


    data_list = []
    for exp in exp_list:
        expobj = learning_curve(exp, runs=20)
        data_list.append(expobj)







    fig, axes = plot_baseline_figure_13(
        data_list=data_list,
        baseline_methods=baseline_methods,
        our_methods=our_methods,
        figsize=(12, 6)
    )
    plt.show()


    # fig, axes = plot_baseline_figure_14(
    #     data_list=data_list,
    #     baseline_methods=baseline_methods,
    #     our_methods=our_methods,
    #     figsize=(12, 4.5)
    # )
    # plt.show()


    plot_para_figure_13(
        data_list=data_list,
        our_methods=our_methods,
        figsize=(12, 5)
    )
    plt.show()
    #
    #
    #
    # table_latex = table_experiment_comparison_only_reward(
    #     data_list=data_list,
    #     baseline_methods=baseline_methods,
    #     our_methods=our_methods,
    #     figsize=(16, 8)
    # )
    # print(table_latex)

    # table_latex = table_experiment_comparison(
    #     data_list=data_list,
    #     baseline_methods=baseline_methods,
    #     our_methods=our_methods,
    #     figsize=(16, 8)
    # )
    # print(table_latex)
    #render_latex_table(table_latex, title='Learning Curve')


    # 可选：保存图形
    # fig.savefig('experiment_results.png', dpi=300, bbox_inches='tight')