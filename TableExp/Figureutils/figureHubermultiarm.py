import math
import os
import re

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast
import numpy as np

class ExperimentData:
    """实验数据类"""

    def __init__(self, env_name, data_dict):
        self.env_name = env_name
        self.data_dict = data_dict

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





def create_allinone_table(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    #methods = list(datalist[0][0][0].data_dict.keys())
    methods = [
        "OrderQ",
        "Q",
        "AveragedQ",




        "MaxminQ",
        "WeightedQ",
        "EBQL",
        "DoubleQ",

        "ECQdelta20.0",
        "ECQdelta40.0",
        "ECQdelta60.0",
        "ECQdelta80.0",
        "ECQdelta100.0",
        "ECQdelta120.0",
        "ECQdelta140.0",

    ]

    table_data = np.zeros((len(methods), 12))
    for main_idx in range(3):
        for row in range(2):
            for col in range(2):
                env_data = datalist[main_idx][row][col]
                for i, method in enumerate(methods):
                    methoddata = env_data.data_dict[method]['Q'][-1]
                    table_data[i, main_idx*4 + row*2 + col] = methoddata #abs(20-methoddata)

    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * 12 + "}")
    latex_code.append(r"\hline ")

    # 表头 - 左上角空白


    cfg1 = r"\multicolumn{4}{c}{$N_{arm}$}"
    cfg2 = r"\multicolumn{4}{c}{$\sigma_{R}$}"
    cfg3 = r"\multicolumn{4}{c}{$\sigma_{Q}$}"
    header = " & ".join([""] + [cfg1, cfg2, cfg3]) + r" \\"

    latex_code.append(header)

    hline = r""
    for i in range(len(data_list)):
        hline += r" \cmidrule(lr){" + str(i * 4 + 2) + "-" + str(i * 4 + 5) + "}"
    latex_code.append(hline)

    col_names = []
    for value_list in settingvalues_list:
        for value in value_list:
            col_names.append(str(value))


    title = " & ".join(["Algorithm"] + col_names) + r" \\"
    latex_code.append(title)

    latex_code.append(r"\hline ")


    # 数据行
    for i, row_name in enumerate(methods):
        if "ECQ" in row_name:
            row_name = f"$\delta={row_name[8:-2]}$"
        row_data = [row_name]

        for j, col_name in enumerate(col_names):
            data = table_data[:, j]
            sorted_idx = np.argsort(abs(20-data))

            if i == sorted_idx[0]:
                row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最小值加粗
            elif len(data) > 1 and i == sorted_idx[1]:
                row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次小值下划线
            else:
                row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    print("\n".join(latex_code))
    return "\n".join(latex_code)



def create_individual_figures_baselines(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    """创建三张单独的图形，每张图包含2x2子图"""
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    #methods = list(datalist[0][0][0].data_dict.keys())
    methods = [
        "Q",
        "MaxminQ",

        "DoubleQ",
        "EBQL",

        "AveragedQ",
        "OrderQ",

        "WeightedQ",


        "M-DQ0.0",
        "M-DQ0.2",
        "M-DQ0.4",
        "M-DQ0.6",
        "M-DQ0.8",
        "M-DQ1.0",

    ]
    mycolor = mycolorclass()

    color_dict = {
        "OrderQ": mycolor.yellow,
        "Q": mycolor.blue,
        "AveragedQ": mycolor.green,
        "MaxminQ": mycolor.molvse,
        "WeightedQ": mycolor.brown,
        "EBQL": mycolor.pink,
        "DoubleQ": mycolor.qianlanse,

        "M-DQ0.0": mycolor.blue,
        "M-DQ0.2": mycolor.green,
        "M-DQ0.4": mycolor.red,
        "M-DQ0.6": mycolor.black,
        "M-DQ0.8": mycolor.yellow,
        "M-DQ1.0": mycolor.pink,


    }

    basewidth = 2
    ourwidth = 2.5
    basestyle = '-'
    ourstyle = '-'



    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    # 小子图标题前缀
    sub_prefix = ["arms", "Rstd", "Qstd"]

    # 存储所有图形的列表
    all_figures = []

    # 为每个大子图创建单独的图形
    for main_idx in range(3):
        # 创建新图形
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 11))
        plt.subplots_adjust(
            left=0.1,  # 左边距 (默认0.125)
            right=0.97,  # 右边距
            bottom=0.1,  # 底边距 (默认0.11)
            top=0.89,  # 顶边距
            wspace=0.2,  # 水平间距 (默认0.2)
            hspace=0.35  # 垂直间距 (默认0.2)
        )

        # 存储当前图形的轴对象


        for row in range(2):
            for col in range(2):
                ax = axes[row][col]

                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict

                # 找出我们方法的最大值
                our_best_method = ""
                bset_maxQ = float("inf")
                for method in methods:
                    if ourmethod_keyword in method:
                        y = sub_data_dict[method]['Q'][-1]
                        if abs(y-20) < bset_maxQ:
                            bset_maxQ = abs(y-20)
                            our_best_method = method

                for method in methods:
                    # 获得每个方法的数据
                    if ourmethod_keyword in method and method == our_best_method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=mycolor.red,
                                linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                                linestyle=basestyle if ourmethod_keyword not in method else ourstyle)


                    elif ourmethod_keyword not in method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=color_dict[method],
                                linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                                linestyle=basestyle if ourmethod_keyword not in method else ourstyle)


                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx = chr(ord('a') + row * 2 + col)
                ax.set_xlabel(f'steps ($t \; \; x100$)\n({charidx}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',
                                  fontsize=20)

                if col == 0:
                    ax.set_ylabel('Maximum Q-value', fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0, 200))

                #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)


        #创建共用图例 - 放在图形中间上方
        #创建图例
        legend_elements = []
        for method in methods:
            if ourmethod_keyword in method:
                continue
            if method != "Q" and method[-1] == "Q":
                if method=="WeightedQ":
                    label = "Weighted DQ"
                else:
                    label = method[:-1]+" Q"
            else:
                label = method
            legend_elements.append(
                plt.Line2D([0], [0], color=color_dict[method],
                           linewidth=3,
                           linestyle='-',
                           label=label)
            )
        legend_elements.append(
            plt.Line2D([0], [0], color=mycolor.red,
                       linewidth=3,
                       linestyle='-',
                       label='DMDQ(Our Best)')
        )
        legend_elements.append(
            plt.Line2D([0], [0], color='black',
                       linewidth=3,
                       linestyle='--',
                       label='unbiased')
        )

        # 添加图例（横排，放在图形中间上方）
        legend = fig.legend(handles=legend_elements,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1),
                            ncol=5,
                            fontsize=17,
                            columnspacing=0.45,
                            handlelength=1.35,
                            frameon=True,
                            fancybox=True,
                            shadow=False,
                            borderpad=0.6)

        all_figures.append(fig)


        fig.savefig(f'../Result/{group_title[main_idx]}2_2.png', dpi=300)

    return all_figures



def create_allinone_figures_baselines(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    methods = list(datalist[0][0][0].data_dict.keys())

    methods = [
        "Q",
        "MaxminQ",

        "DoubleQ",
        "EBQL",

        "AveragedQ",
        "OrderQ",

        "WeightedQ",


        "ECQdelta20.0",
        "ECQdelta40.0",
        "ECQdelta60.0",
        "ECQdelta80.0",
        "ECQdelta100.0",
        "ECQdelta120.0",
        "ECQdelta140.0",

    ]



    mycolor = mycolorclass()

    color_dict = {
        "OrderQ": mycolor.brown,
        "Q": mycolor.blue,
        "AveragedQ": mycolor.green,
        #"EnsembleQ": mycolor.yellow,
        "MaxminQ": mycolor.qianlanse,
        "WeightedQ": mycolor.black,
        "EBQL": mycolor.molvse,
        "DoubleQ": mycolor.pink,
        "ACCDQ": mycolor.yellow,

    }

    methos2lable = {
        "OrderQ": "Order Q",
        "Q": "Q",
        "AveragedQ": "Averaged Q",
        # "EnsembleQ": mycolor.yellow,
        "MaxminQ": "Maxmin Q",
        "WeightedQ": "Weighted Q",
        "EBQL": "EBQL",
        "DoubleQ": "Double Q",
        "ACCDQ": "AC-CDQ",

    }

    basewidth = 2
    ourwidth = 2
    basestyle = '-'
    ourstyle = '-'



    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    # 小子图标题前缀
    sub_prefix = ["Narm", "Rstd", "Qstd"]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))
    plt.subplots_adjust(
        left=0.05,  # 左边距 (默认0.125)
        right=0.985,  # 右边距
        bottom=0.07,  # 底边距 (默认0.11)
        top=0.92,  # 顶边距
        wspace=0.2,  # 水平间距 (默认0.2)
        hspace=0.32  # 垂直间距 (默认0.2)
    )

    # 为每个大子图创建单独的图形
    for main_idx in range(3):
        # 创建新图形
        for row in range(2):
            for col in range(2):
                ax = axes[main_idx, row*2+col]

                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict

                # 找出我们方法的最大值
                our_best_method = ""
                bset_maxQ = float("inf")
                for method in methods:
                    if ourmethod_keyword in method:
                        y = sub_data_dict[method]['Q'][-1]
                        if abs(y-20) < bset_maxQ:
                            bset_maxQ = abs(y-20)
                            our_best_method = method

                for method in methods:
                    # 获得每个方法的数据
                    if ourmethod_keyword in method and method == our_best_method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=mycolor.red,
                                linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                                linestyle=basestyle if ourmethod_keyword not in method else ourstyle)


                    elif ourmethod_keyword not in method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=color_dict[method],
                                linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                                linestyle=basestyle if ourmethod_keyword not in method else ourstyle)


                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx = chr(ord('A') + main_idx)
                ax.set_xlabel(f'Steps $(t x 100)$\n{charidx}({int(row*2+col+1):d}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',fontsize=20)

                if row==0 and col == 0:
                    ax.set_ylabel('Maximum Q-value', fontsize=20)
                # if main_idx == 1:
                #     ax.set_xlabel(f'Steps $(t x 100)$\n{charidx}({int(row * 2 + col + 1):d}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0, 100))

                #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)


        #创建共用图例 - 放在图形中间上方
        #创建图例
        legend_elements = []
        for method in methods:
            if ourmethod_keyword in method:
                continue
            legend_elements.append(
                plt.Line2D([0], [0], color=color_dict[method],
                           linewidth=3,
                           linestyle='-',
                           #label=method
                           label=methos2lable[method],
                           )
            )
        legend_elements.append(
            plt.Line2D([0], [0], color=mycolor.red,
                       linewidth=3,
                       linestyle='-',
                       label='Huber Q(Our Best)')
        )
        legend_elements.append(
            plt.Line2D([0], [0], color='black',
                       linewidth=3,
                       linestyle='--',
                       label='unbiased')
        )

        # 添加图例（横排，放在图形中间上方）
        legend = fig.legend(handles=legend_elements,
                            handlelength=2.5,
                            handletextpad=0.6,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1),
                            columnspacing=8.3,
                            ncol=5,
                            fontsize=18,
                            frameon=True,
                            fancybox=True,
                            shadow=False,
                            borderpad=0.8)




    fig.savefig(f'../Result/tablebaseline3_4.png', dpi=300)

    return

def create_allinone_figures_baselines2(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    methods = list(datalist[0][0][0].data_dict.keys())

    methods = [
        "Q",
        "MaxminQ",

        "DoubleQ",
        "EBQL",

        "AveragedQ",
        "OrderQ",

        "WeightedQ",


        "ECQdelta20.0",
        "ECQdelta40.0",
        "ECQdelta60.0",
        "ECQdelta80.0",
        "ECQdelta100.0",
        "ECQdelta120.0",
        "ECQdelta140.0",

    ]



    mycolor = mycolorclass()

    color_dict = {
        "OrderQ": mycolor.brown,
        "Q": mycolor.blue,
        "AveragedQ": mycolor.green,
        #"EnsembleQ": mycolor.yellow,
        "MaxminQ": mycolor.qianlanse,
        "WeightedQ": mycolor.black,
        "EBQL": mycolor.molvse,
        "DoubleQ": mycolor.pink,
        "ACCDQ": mycolor.yellow,

    }

    methos2lable = {
        "OrderQ": "Order Q",
        "Q": "Q",
        "AveragedQ": "Averaged Q",
        # "EnsembleQ": mycolor.yellow,
        "MaxminQ": "Maxmin Q",
        "WeightedQ": "Weighted Q",
        "EBQL": "EBQL",
        "DoubleQ": "Double Q",
        "ACCDQ": "AC-CDQ",

    }

    basewidth = 3
    ourwidth = 3
    basestyle = '-'
    ourstyle = '-'



    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    # 小子图标题前缀
    sub_prefix = ["Narm", "Rstd", "Qstd"]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    plt.subplots_adjust(
        left=0.05,  # 左边距 (默认0.125)
        right=0.985,  # 右边距
        bottom=0.11,  # 底边距 (默认0.11)
        top=0.89,  # 顶边距
        wspace=0.2,  # 水平间距 (默认0.2)
        hspace=0.21  # 垂直间距 (默认0.2)
    )

    # 为每个大子图创建单独的图形
    for main_idx in range(2):
        # 创建新图形
        for row in range(2):
            for col in range(2):
                ax = axes[main_idx, row*2+col]

                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict

                # 找出我们方法的最大值
                our_best_method = ""
                bset_maxQ = float("inf")
                for method in methods:
                    if ourmethod_keyword in method:
                        y = sub_data_dict[method]['Q'][-1]
                        if abs(y-20) < bset_maxQ:
                            bset_maxQ = abs(y-20)
                            our_best_method = method

                for method in methods:
                    # 获得每个方法的数据
                    if ourmethod_keyword in method and method == our_best_method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=mycolor.red,
                                linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                                linestyle=basestyle if ourmethod_keyword not in method else ourstyle)


                    elif ourmethod_keyword not in method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=color_dict[method],
                                linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                                linestyle=basestyle if ourmethod_keyword not in method else ourstyle)


                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased", linewidth=2)

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx = chr(ord('A') + main_idx)
                #ax.set_xlabel(f'Steps $(t x 100)$\n{charidx}({int(row*2+col+1):d}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',fontsize=20)
                ax.set_xlabel(
                    f'{charidx}({int(row * 2 + col + 1):d}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',
                    fontsize=30)

                if row==0 and col == 0:
                    ax.set_ylabel('Maximum Q-value', fontsize=30)
                if main_idx == 1:
                    ax.set_xlabel(f'Steps $(t x 100)$\n{charidx}({int(row * 2 + col + 1):d}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',fontsize=30)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0, 100))

                #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)


        #创建共用图例 - 放在图形中间上方
        #创建图例
        legend_elements = []
        for method in methods:
            if ourmethod_keyword in method:
                continue
            legend_elements.append(
                plt.Line2D([0], [0], color=color_dict[method],
                           linewidth=3,
                           linestyle='-',
                           #label=method
                           label=methos2lable[method],
                           )
            )
        legend_elements.append(
            plt.Line2D([0], [0], color=mycolor.red,
                       linewidth=3,
                       linestyle='-',
                       label='Huber Q(Our Best)')
        )
        legend_elements.append(
            plt.Line2D([0], [0], color='black',
                       linewidth=3,
                       linestyle='--',
                       label='unbiased')
        )

        # 添加图例（横排，放在图形中间上方）
        legend = fig.legend(handles=legend_elements,
                            handlelength=2.5,
                            handletextpad=0.4,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1.01),
                            columnspacing=3,
                            ncol=5,
                            fontsize=25,
                            frameon=True,
                            fancybox=True,
                            shadow=False,
                            borderpad=0.4)




    fig.savefig(f'../Result/tablebaseline2_4.png', dpi=300)

    return


def create_allinone_figures_pararho(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    """创建三张单独的图形，每张图包含2x2子图"""
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    #methods = list(datalist[0][0][0].data_dict.keys())
    methods = [
        "ECQdelta20.0",
        "ECQdelta40.0",
        "ECQdelta60.0",
        "ECQdelta80.0",
        "ECQdelta100.0",
        "ECQdelta120.0",
        "ECQdelta140.0",
    ]
    mycolor = mycolorclass()

    color_dict = {

        "ECQdelta20.0": mycolor.blue,
        "ECQdelta40.0": mycolor.green,
        "ECQdelta60.0": mycolor.molvse,
        "ECQdelta80.0": mycolor.red,
        "ECQdelta100.0": mycolor.qianlanse,
        "ECQdelta120.0": mycolor.pink,
        "ECQdelta140.0": mycolor.yellow,
        #"ECQdelta0.0": mycolor.black,


    }

    basewidth = 2
    ourwidth = 2
    basestyle = '-'
    ourstyle = '-'



    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    # 小子图标题前缀
    sub_prefix = ["Narm", "Rstd", "Qstd"]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    plt.subplots_adjust(
        left=0.06,  # 左边距 (默认0.125)
        right=0.985,  # 右边距
        bottom=0.07,  # 底边距 (默认0.11)
        top=0.94,  # 顶边距
        wspace=0.2,  # 水平间距 (默认0.2)
        hspace=0.32  # 垂直间距 (默认0.2)
    )

    # 为每个大子图创建单独的图形
    for main_idx in range(3):
        for row in range(2):
            for col in range(2):
                ax = axes[main_idx, row*2+col]

                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict



                for method in methods:
                    # 获得每个方法的数据

                    y = sub_data_dict[method]['Q']
                    x = [i for i in range(len(y))]
                    ax.plot(x, y, color=color_dict[method],
                            linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                            #label='M-DQ',
                            linestyle=basestyle if ourmethod_keyword not in method else ourstyle)
                    ax.plot([], [], color=color_dict[method],
                            linewidth=3,
                            label=method,
                            linestyle='-')

                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx = chr(ord('A') + main_idx)
                ax.set_xlabel(f'Steps $(t x 100)$\n{charidx}({int(row*2+col+1):d}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',
                                  fontsize=20)

                if col == 0 and row==0:
                    ax.set_ylabel('Maximum Q-value', fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0, 100))
                #ax.set_ylim((10, 35))

                #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

        #创建共用图例 - 放在图形中间上方
        #创建图例
        legend_elements = []
        for method in methods:

            legend_elements.append(
                plt.Line2D([0], [0], color=color_dict[method],
                           linewidth=3,
                           linestyle='-',
                           label=r"$\delta$="+method[8:-2])
            )

        legend_elements.append(
            plt.Line2D([0], [0], color='black',
                       linewidth=3,
                       linestyle='--',
                       label='unbiased')
        )

        # 添加图例（横排，放在图形中间上方）
        legend = fig.legend(handles=legend_elements,
                            handlelength=2.5,
                            handletextpad=0.6,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1),
                            columnspacing=4,
                            ncol=9,
                            fontsize=18,
                            frameon=True,
                            fancybox=True,
                            shadow=False,
                            borderpad=0.8)



    fig.savefig(f'../Result/tablepararho3_4.png', dpi=300)

    return


def create_individual_figures_pararho(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    """创建三张单独的图形，每张图包含2x2子图"""
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    #methods = list(datalist[0][0][0].data_dict.keys())
    methods = [
        "M-DQ0.0",
        "M-DQ0.5",

        "M-DQ0.2",
        "M-DQ0.6",

        "M-DQ0.3",
        "M-DQ0.8",

        "M-DQ0.4",
        "M-DQ1.0",
    ]
    mycolor = mycolorclass()

    # color_dict = {
    #
    #     "M-DQ0.0": mycolor.blue,
    #     "M-DQ0.2": mycolor.green,
    #     "M-DQ0.3": mycolor.molvse,
    #     "M-DQ0.4": mycolor.red,
    #     "M-DQ0.5": mycolor.qianlanse,
    #     "M-DQ0.6": mycolor.black,
    #     "M-DQ0.8": mycolor.yellow,
    #     "M-DQ1.0": mycolor.pink,
    #
    #
    # }
    mycolor.blue,
    mycolor.molvse,
    mycolor.qianlanse,
    mycolor.pink,
    mycolor.green,
    mycolor.yellow,
    color_dict = {
        "M-DQ0.0": mycolor.blue,
        "M-DQ0.5": mycolor.molvse,

        "M-DQ0.2":mycolor.qianlanse,
        "M-DQ0.6":mycolor.green,

        "M-DQ0.3":mycolor.pink,
        "M-DQ0.8":mycolor.yellow,

        "M-DQ0.4":mycolor.red,
        "M-DQ1.0":mycolor.brown,


    }

    basewidth = 2
    ourwidth = 2
    basestyle = '-'
    ourstyle = '-'



    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    # 小子图标题前缀
    sub_prefix = ["arms", "Rstd", "Qstd"]

    # 存储所有图形的列表
    all_figures = []

    # 为每个大子图创建单独的图形
    for main_idx in range(3):
        # 创建新图形
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 11))
        plt.subplots_adjust(
            left=0.1,  # 左边距 (默认0.125)
            right=0.97,  # 右边距
            bottom=0.1,  # 底边距 (默认0.11)
            top=0.89,  # 顶边距
            wspace=0.2,  # 水平间距 (默认0.2)
            hspace=0.35  # 垂直间距 (默认0.2)
        )

        # 存储当前图形的轴对象


        for row in range(2):
            for col in range(2):
                ax = axes[row][col]

                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict

                for method in methods:
                    # 获得每个方法的数据

                    y = sub_data_dict[method]['Q']
                    x = [i for i in range(len(y))]
                    ax.plot(x, y, color=color_dict[method],
                            linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                            linestyle=basestyle if ourmethod_keyword not in method else ourstyle)

                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx = chr(ord('a') + row * 2 + col)
                ax.set_xlabel(f'steps ($t \; \; x100$)\n({charidx}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}',
                                  fontsize=20)

                if col == 0:
                    ax.set_ylabel('Maximum Q-value', fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0, 200))

                #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

        #创建共用图例 - 放在图形中间上方
        #创建图例
        legend_elements = []
        for method in methods:
            label = r'$\rho='+method[4:]+"$"
            legend_elements.append(
                plt.Line2D([0], [0], color=color_dict[method],
                           linewidth=3,
                           linestyle='-',
                           label=label)
            )

        legend_elements.append(
            plt.Line2D([0], [0], color='black',
                       linewidth=3,
                       linestyle='--',
                       label='unbiased')
        )

        # 添加图例（横排，放在图形中间上方）
        legend = fig.legend(handles=legend_elements,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1),
                            ncol=5,
                            fontsize=17,
                            handlelength=1.7,
                            columnspacing=2.4,
                            frameon=True,
                            fancybox=True,
                            shadow=False,
                            borderpad=0.7)

        all_figures.append(fig)

        fig.savefig(f'../Result/{group_title[main_idx]}pararho2_2.png', dpi=300)

    return all_figures





def create_appendix_figure(datalist, settingvalues_list, ourmethod_keyword = "mix"):

    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    methods = [
        "OrderQ",
        "Q",
        "AveragedQ",
        "MaxminQ",
        "WeightedQ",
        "EBQL",
        "DoubleQ",


        "M-DQ1.0",
        "M-DQ0.9",
        "M-DQ0.8",
        "M-DQ0.7",
        "M-DQ0.6",
        "M-DQ0.5",
        "M-DQ0.4",
        "M-DQ0.3",
        "M-DQ0.2",
        "M-DQ0.1",
        "M-DQ0.0",
    ]

    mycolor = mycolorclass()

    color_dict = {
        "OrderQ": mycolor.black,
        "Q": mycolor.blue,
        "AveragedQ": mycolor.green,
        "EnsembleQ": mycolor.yellow,
        "MaxminQ": mycolor.pink,
        "WeightedQ": mycolor.brown,
        "EBQL": mycolor.molvse,
        "DoubleQ": mycolor.qianlanse,

        "M-DQ0.0": mycolor.black,
        "M-DQ0.1": mycolor.blue,
        "M-DQ0.2": mycolor.green,
        "M-DQ0.3": mycolor.brown,
        "M-DQ0.4": mycolor.red,
        "M-DQ0.5": mycolor.molvse,
        "M-DQ0.6": mycolor.qianlanse,
        "M-DQ0.7": mycolor.purple,
        "M-DQ0.8": mycolor.yellow,
        "M-DQ0.9": mycolor.lianghuangse,
        "M-DQ1.0": mycolor.pink,


    }

    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    #小子图标题前缀
    sub_prefix = ["Narm", "Rstd", "Qstd"]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    plt.subplots_adjust(
        left=0.08,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.1,  # 底边距 (默认0.11)
        top=0.99,  # 顶边距
        wspace=0.3,  # 水平间距 (默认0.2)
        hspace=0.4  # 垂直间距 (默认0.2)
    )

    # baseline
    for main_idx in range(3):
        for row in range(1):
            for col in range(4):
                ax = axes[main_idx][col]


                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict

                # 找出我们方法的最大值
                our_best_method = ""
                bset_maxQ = float("inf")
                for method in methods:
                    if ourmethod_keyword in method:
                        y = sub_data_dict[method]['Q'][-1]
                        if abs(y-20) < bset_maxQ:
                            bset_maxQ = abs(y-20)
                            our_best_method = method

                for method in methods:
                    if our_best_method == method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=mycolor.red,
                                linewidth=2,
                                label=method,
                                linestyle='-')
                    elif ourmethod_keyword not in method:
                        y = sub_data_dict[method]['Q']
                        x = [i for i in range(len(y))]
                        ax.plot(x, y, color=color_dict[method],
                                linewidth=2,
                                label=method,
                                linestyle='-')

                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx =  chr(ord('a') + main_idx*4 + col)

                ax.set_xlabel(f'steps(x100)\n({charidx}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}', fontsize=20)

                if col == 0:
                    ax.set_ylabel('MaxQ', fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0,200))

                ax.legend(fontsize=14, loc=[0,0],handlelength=1, framealpha=0.5)

    fig.savefig(f'../Result/appendixbaseline.png', dpi=300)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    plt.subplots_adjust(
        left=0.08,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.1,  # 底边距 (默认0.11)
        top=0.99,  # 顶边距
        wspace=0.3,  # 水平间距 (默认0.2)
        hspace=0.4  # 垂直间距 (默认0.2)
    )
    #rho
    for main_idx in range(3):
        for row in range(1):
            for col in range(4):
                ax = axes[main_idx][col]


                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict
                for method in methods:
                    # 获得每个方法的数据

                    if ourmethod_keyword not in method:
                        continue
                    y = sub_data_dict[method]['Q']
                    x = [i for i in range(len(y))]
                    ax.plot(x, y, color=color_dict[method],
                            linewidth=2,
                            label=method,
                            linestyle='-')

                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx =  chr(ord('a') + main_idx*4 + col)

                ax.set_xlabel(f'steps(x100)\n({charidx}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}', fontsize=20)
                if col == 0:
                    ax.set_ylabel('MaxQ', fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0,200))

                ax.legend(fontsize=14, loc=[0,0],handlelength=1, framealpha=0.5)
    fig.savefig(f'../Result/appendixrho.png', dpi=300)

    plt.show()



    return fig, axes

def create_figure(datalist, settingvalues_list, ourmethod_keyword = "mix"):
    """创建包含三个大子图的图形，每个大子图内有2x2小子图"""
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 方法名称和颜色
    # 方法名称和颜色
    methods = list(datalist[0][0][0].data_dict.keys())

    color_dict = {}

    baselinecolors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    ourcolors = plt.cm.tab20b(np.linspace(0, 1, len(methods)))
    baseindex = 0
    ourindex = 0
    basewidth = 2.5
    ourwidth = 2.5
    basestyle = '-'
    ourstyle = '-'
    for method in methods:
        if ourmethod_keyword in method:
            color_dict[method] = ourcolors[ourindex]
            ourindex += 1
        else:
            color_dict[method] = baselinecolors[baseindex]
            baseindex += 1


    mycolor = mycolorclass()

    color_dict = {
        "OrderQ": mycolor.black,
        "Q": mycolor.blue,
        "AveragedQ": mycolor.green,
        "EnsembleQ": mycolor.yellow,
        "MaxminQ": mycolor.pink,
        "WeightedQ": mycolor.brown,
        "EBQL": mycolor.molvse,
        "DoubleQ": mycolor.qianlanse,

        "M-DQ0.0": mycolor.blue,
        "M-DQ0.2": mycolor.green,
        "M-DQ0.4": mycolor.red,
        "M-DQ0.6": mycolor.black,
        "M-DQ0.8": mycolor.yellow,
        "M-DQ1.0": mycolor.pink,


    }

    # 大子图标题
    group_title = ["Narm", "Rstd", "Qstd"]

    #小子图标题前缀
    sub_prefix = ["Narm", "Rstd", "Qstd"]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    plt.subplots_adjust(
        left=0.08,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.1,  # 底边距 (默认0.11)
        top=0.99,  # 顶边距
        wspace=0.3,  # 水平间距 (默认0.2)
        hspace=0.4  # 垂直间距 (默认0.2)
    )

    # 为每个大子图创建1x4的小子图
    for main_idx in range(3):
        for row in range(1):
            for col in range(4):
                ax = axes[main_idx][col]


                # 获得当前子图的数据
                sub_data_dict = datalist[main_idx][row][col].data_dict
                for method in methods:
                    # 获得每个方法的数据
                    if col != 3 and ourmethod_keyword in method:
                        if "0.4" not in method:
                            continue
                    if col==3 and ourmethod_keyword not in method:
                        continue
                    y = sub_data_dict[method]['Q']
                    x = [i for i in range(len(y))]
                    ax.plot(x, y, color=color_dict[method],
                            linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
                            label=method,
                            linestyle=basestyle if ourmethod_keyword not in method else ourstyle)

                # 添加网格
                ax.grid(True, alpha=1, linestyle='-')
                ax.axhline(y=20, color='black', linestyle='--', label="Unbiased")

                # 设置轴标签
                value_list = settingvalues_list[main_idx]
                charidx =  chr(ord('a') + main_idx*4 + col)
                if col != 3:
                    ax.set_xlabel(f'steps(x100)\n({charidx}) {sub_prefix[main_idx]}={value_list[row * 2 + col]}', fontsize=20)
                else:
                    ax.set_xlabel(f'steps(x100)\n({charidx})'+r"""effect of $\rho$""", fontsize=20)
                if col == 0:
                    ax.set_ylabel('MaxQ', fontsize=20)

                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim((0,200))

                ax.legend(fontsize=14, loc=[0,0],handlelength=1, framealpha=0.5)



    # 在大子图上方添加标题
    # for i, title in enumerate(group_title):
    #     bbox = gs_main[i].get_position(fig)
    #     fig.text(bbox.x0 + (bbox.x1 - bbox.x0) / 2, group_top+0.03, title,
    #              ha='center', va='top', fontsize=12, fontweight='bold')

    # 创建共用图例 - 放在图形中间上方
    # 创建图例
    # legend_elements = []
    # for method in methods:
    #     legend_elements.append(
    #         plt.Line2D([0], [0], color=color_dict[method],
    #                    linewidth=basewidth if ourmethod_keyword not in method else ourwidth,
    #                    linestyle=basestyle if ourmethod_keyword not in method else ourstyle,
    #                    label=method)
    #     )
    #
    # # 添加图例（横排，放在图形中间上方）
    # legend = fig.legend(handles=legend_elements,
    #                     loc='upper center',
    #                     bbox_to_anchor=(0.5, 1),
    #                     ncol=len(methods)//2 if len(methods)%2==0 else len(methods)//2+1,
    #                     fontsize=17,
    #                     frameon=True,
    #                     fancybox=True,
    #                     shadow=False,
    #                     borderpad=0.8)


    plt.show()

    fig.savefig(f'../Result/tabres.png', dpi=300)

    return fig, axes


def create_latex_table(row_names, data_dict, title="table", label="tab:my_table"):
    """
    生成LaTeX三线表代码
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
                afterfix=r"\%"
            else:
                afterfix=""
            if i == sorted_idx[0]:
                row_data.append(fr"\textbf{{{data[i]:.2f}}}"+afterfix)  # 最大值加粗
            elif len(data) > 1 and i == sorted_idx[1]:
                row_data.append(fr"\underline{{{data[i]:.2f}}}"+afterfix)  # 次大值下划线
            else:
                row_data.append(f"{data[i]:.2f}"+afterfix)

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)



def render_latex_table(table_tex_code):

    # 创建完整的LaTeX文档
    print(table_tex_code)
    full_latex_code = table_tex_code.replace("\n", " ")


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        'figure.dpi': 200,
    })

    fig, ax = plt.subplots(figsize=(3.5, 5))
    ax.axis('off')

    # 渲染LaTeX表格
    ax.text(0.5, 0.5, full_latex_code, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            horizontalalignment='center')

    #plt.tight_layout()
    plt.show()



def tableQ(multiarm, labels, title="Q", V_star=0):

    for i in range(len(multiarm)):
        multiarm[i] = multiarm[i][-1]

    row_names = labels


    multiarm = np.array(multiarm).flatten()
    Q_col = multiarm
    absMaxQ = np.abs(Q_col-V_star)
    improve_col = (absMaxQ[0] - absMaxQ) / absMaxQ[0]
    data_dict = {"MaxQ":Q_col,
                 "abserrMaxQ": absMaxQ,
                 "improve": improve_col}



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


            if col_name == "MaxQ":
                row_data.append(f"{data[i]:.2f}")
                continue

            if "abs" in col_name:
                sorted_idx = np.argsort(data)
            else:
                sorted_idx = np.argsort(data)[::-1]


            if col_name == "improve":
                afterfix = r"\%"
            else:
                afterfix = ""



            if i == sorted_idx[0]:
                row_data.append(fr"\textbf{{{data[i]:.2f}}}" + afterfix)  # 最值加粗
            elif len(data) > 1 and i == sorted_idx[1]:
                row_data.append(fr"\underline{{{data[i]:.2f}}}" + afterfix)  # 次最值下划线
            else:
                row_data.append(f"{data[i]:.2f}" + afterfix)

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    latex_code = "\n".join(latex_code)

    #latex_code = create_latex_table(row_names, data_dict, title=title)
    #test_render_latex_table()
    render_latex_table(latex_code)

def tablePrglobal(multiarm, labels, title="Pr"):
    for i in range(len(multiarm)):
        multiarm[i] = multiarm[i][-1]

    row_names = labels

    multiarm = np.array(multiarm).flatten()
    Pr_col = multiarm
    improve_col = (multiarm - multiarm[0]) / multiarm[0]
    data_dict = {"Pr": Pr_col,
                 "improve": improve_col}

    latex_code = create_latex_table(row_names, data_dict, title=title)
    # test_render_latex_table()
    render_latex_table(latex_code)

def tableReward(multiarm, labels, title="R"):
    for i in range(len(multiarm)):
        multiarm[i] = multiarm[i][-1]

    row_names = labels

    multiarm = np.array(multiarm).flatten()
    R_col = multiarm
    improve_col = (multiarm - multiarm[0]) / np.abs(multiarm[0])
    data_dict = {"Reward": R_col,
                 "improve": improve_col}

    latex_code = create_latex_table(row_names, data_dict, title=title)
    # test_render_latex_table()
    render_latex_table(latex_code)

def moving_average(data, window_size):
    """移动平均平滑"""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')



def get_Q_value(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-5][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S

def get_Pr_sa(dir):
    log = open(dir, 'r').readlines()
    Pr_sa = log[-3][:]
    Pr_sa = ast.literal_eval(Pr_sa)
    return Pr_sa

def get_r_perstep(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-1][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S

def cfg2envdata(cfgdir, cfgname, algorithms, Klist):
    data_len = 200

    pngname = cfgname

    Result_path = os.path.join(cfgdir, cfgname)
    multiarmQ = []
    multiarmPr = []

    multiarmR = []
    labels = []
    for algorithm in os.listdir(Result_path):
        if algorithm not in algorithms:
            continue
        algorithm_dir = os.path.join(Result_path, algorithm)
        if not os.path.isdir(algorithm_dir):
            continue




        if algorithm in ['AveragedQ', "EnsembleQ", "EBQL", "KQ"]:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+K(\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字

                    K = float(match.group(1))
                    if K not in Klist:
                        continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    #labels.append(f'{algorithm}K{K}')
                    labels.append(f'{algorithm}')

        elif algorithm in ['WeightedQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+c(\d+\.\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    c = float(match.group(1))

                    if c != 10:
                        continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    #labels.append(f'{algorithm}c{c}')
                    labels.append(f'{algorithm}')


        elif algorithm in ['AdaOQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+M(\d+)m(\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    M = int(match.group(1))
                    m = int(match.group(2))
                    # if m*2 != M or M not in Klist:
                    #     continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    #labels.append(f'{algorithm}M{M}m{m}')
                    #labels.append(f'OrderQM{M}m{m}')
                    labels.append(f'OrderQ')
        elif algorithm in ['ACCDQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+mr(\d+\.\d+)K(\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    pQ = float(match.group(1))
                    K = int(match.group(2))
                    # if pQ != 0.4:
                    #     continue
                    if K != 2:
                        continue

                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    # labels.append(f'ACC{K:d}')
                    labels.append(f'ACCDQ')
        elif algorithm in ['ECQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+delta(\d+\.\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    delta = float(match.group(1))

                    # if c != 10:
                    #     continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    labels.append(f'{algorithm}delta{delta}')


        else:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Q*'
                match = re.search(pattern, filename)
                if match:
                    # if int(match.group(1)) < 15:
                    #    continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])
                    if algorithm == 'SingleQ':
                        labels.append("Q")
                    elif algorithm == 'DoubleQ':
                        labels.append("DoubleQ")
                    else:
                        labels.append(algorithm)


    if "arm" in cfgname:
        env_name = "Multiarm"
        env_data_dict = {}
        for i in range(len(labels)):
            baselinename = labels[i]
            baselinedata_dict = {
                "Q": multiarmQ[i],
            }
            env_data_dict[baselinename] = baselinedata_dict
    elif "huber" in cfgname:
        env_name = "HuberMultiarm"
        env_data_dict = {}
        for i in range(len(labels)):
            baselinename = labels[i]
            baselinedata_dict = {
                "Q": multiarmQ[i],
            }
            env_data_dict[baselinename] = baselinedata_dict
    else:
        raise NotImplementedError

    env_data = ExperimentData(env_name, env_data_dict)
    return env_data

if __name__ == "__main__":

    cfgdir = os.path.join("..", "Result")


    defaultQstd = '1.0'
    defaultQmean = '20.0'
    defaultarmstd = '1.0'
    defaultNarm = '10'
    cfgnamedict = {
        (0, 0, 0): f"huberenvProb0.01Na5as{defaultarmstd}Qm{defaultQmean}Qs{defaultQstd}",
        (0, 0, 1): f"huberenvProb0.01Na10as{defaultarmstd}Qm{defaultQmean}Qs{defaultQstd}",
        (0, 1, 0): f"huberenvProb0.01Na15as{defaultarmstd}Qm{defaultQmean}Qs{defaultQstd}",
        (0, 1, 1): f"huberenvProb0.01Na20as{defaultarmstd}Qm{defaultQmean}Qs{defaultQstd}",


        (1, 0, 0): f"huberenvProb0.01Na{defaultNarm}as1.0Qm{defaultQmean}Qs{defaultQstd}",
        (1, 0, 1): f"huberenvProb0.01Na{defaultNarm}as2.0Qm{defaultQmean}Qs{defaultQstd}",
        (1, 1, 0): f"huberenvProb0.01Na{defaultNarm}as4.0Qm{defaultQmean}Qs{defaultQstd}",
        (1, 1, 1): f"huberenvProb0.01Na{defaultNarm}as8.0Qm{defaultQmean}Qs{defaultQstd}",


        (2, 0, 0): f"huberenvProb0.01Na{defaultNarm}as{defaultarmstd}Qm{defaultQmean}Qs1.0",
        (2, 0, 1): f"huberenvProb0.01Na{defaultNarm}as{defaultarmstd}Qm{defaultQmean}Qs2.0",
        (2, 1, 0): f"huberenvProb0.01Na{defaultNarm}as{defaultarmstd}Qm{defaultQmean}Qs3.0",
        (2, 1, 1): f"huberenvProb0.01Na{defaultNarm}as{defaultarmstd}Qm{defaultQmean}Qs4.0"
    }
    data_list = [[[None for _ in range(2)] for _ in range(2)] for _ in range(3)]


    baseline_methods = []
    our_methods = []
    ourmethod_keyword = "ECQ"
    for key in cfgnamedict:
        i, j ,k = key
        kcfg = [[2]]

        cfgname = cfgnamedict[key]
        for Klist in kcfg:
            algorithmlist = ["ECQ", "SingleQ", "DoubleQ", "AveragedQ", "MaxminQ", "WeightedQ",
                             "EBQL", "ACCDQ","AdaOQ"]

            envdata = cfg2envdata(cfgdir, cfgname, algorithmlist, Klist)
            data_list[i][j][k] = envdata


    create_allinone_figures_baselines(data_list, [[5, 10, 15, 20], [1, 2, 4, 8], [1, 2, 3, 4]], ourmethod_keyword)
    #create_allinone_figures_pararho(data_list, [[5, 10, 15, 20], [1, 2, 4, 8], [1, 2, 3, 4]], ourmethod_keyword)
    #create_allinone_table(data_list, [[5, 10, 15, 20], [1, 2, 4, 8], [1, 2, 3, 4]], ourmethod_keyword)
    plt.show()

