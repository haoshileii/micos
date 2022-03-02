import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from . import _eval_protocols as eval_protocols
# 动态调整上下、左右翻转角度
import sklearn
import torch
# Random state.
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)
RS = 20150101
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from sklearn.decomposition import PCA
from ipywidgets import interact
import numpy as np
import mglearn
import pandas as pd
# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
def draw_svm(test_x_rep, test_y):
#X,y = make_circles(100,noise=0.1,factor=0.5,random_state=100)
    X = test_x_rep
    X = PCA(2).fit_transform(X)
    y = test_y

    #有子图
    #kernel = ['rbf']
    #fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,20))
    # for i,core in enumerate(kernel,1):
    #     xx,yy = ((min(X[:,0]-1),max(X[:,0]+1)),(min(X[:,1]-1),max(X[:,1]+1)))
    #     xx = np.arange(xx[0],xx[1],step=0.1)
    #     yy = np.arange(yy[0],yy[1],step=0.1)
    #     XX,YY = np.meshgrid(xx,yy)
    #     grid = np.c_[XX.ravel(),YY.ravel()]
    #     # 预测类别
    #     #model = SVC(kernel=core,gamma="scale",decision_function_shape="ovo",degree=3,C=1.0)
    #     # model = SVC(kernel=core,gamma="scale",decision_function_shape="ovr",degree=3,C=1.0)
    #     # model.fit(X,y)
    #     model = eval_protocols.fit_svm(X, y)
    #     score=model.score(X,y)
    #     prediction = model.predict(grid).reshape(XX.shape)
    #     # if i-1==0:
    #     #     ax[0].set_title("raw data",fontsize=20)
    #     #     ax[0].scatter(X[:,0],X[:,1],c=y,s=40)
    #     ax[i].set_title(core,fontsize=20)
    #     ax[i].scatter(X[:,0],X[:,1],c=y,label="%.2f"% score,s=40,zorder=10)
    #     ax[i].scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=70,edgecolor="red",facecolor="none",zorder=10)
    #     ax[i].pcolormesh(XX,YY,prediction,cmap=plt.cm.Accent,shading="auto")
    #     ax[i].contour(XX,YY,prediction,colors=["pink","blue","pink"],linestyles=['dashed','solid','dashed'])
    #     ax[i].legend(loc=4)
    #有子图

    core = 'rbf'
    xx, yy = ((min(X[:, 0] - 1), max(X[:, 0] + 1)), (min(X[:, 1] - 1), max(X[:, 1] + 1)))
    xx = np.arange(xx[0], xx[1], step=0.1)
    yy = np.arange(yy[0], yy[1], step=0.1)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    # 预测类别
    # model = SVC(kernel=core,gamma="scale",decision_function_shape="ovo",degree=3,C=1.0)
    # model = SVC(kernel=core,gamma="scale",decision_function_shape="ovr",degree=3,C=1.0)
    # model.fit(X,y)
    model = eval_protocols.fit_svm(X, y)
    score = model.score(X, y)
    prediction = model.predict(grid).reshape(XX.shape)
    # if i-1==0:
    #     ax[0].set_title("raw data",fontsize=20)
    #     ax[0].scatter(X[:,0],X[:,1],c=y,s=40)
    plt.title("classification", fontsize=20)
    plt.scatter(X[:, 0], X[:, 1], c=y, label="%.2f" % score, s=250, zorder=10)
    # plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=70, edgecolor="red", facecolor="none",
    #               zorder=10)
    cm_light = mpl.colors.ListedColormap(['#f4f0e6', '#d9d9f3','#ceefe4','#9dd3a8'])#由于3分类，设置3种颜色或者多种
    plt.pcolormesh(XX, YY, prediction, cmap=cm_light, shading="auto")
    plt.contour(XX, YY, prediction, colors=["k"], linestyles=['solid'])
    plt.xticks([])
    plt.yticks([])
    #加上图例
    # plt.legend(loc=3)
    plt.savefig('results/SVMA.svg')
    plt.show()
def draw_tsne_2D(test_x_rep, test_y, sne):
    X = torch.from_numpy(test_x_rep)
    X_num = X.size(0)
    X = test_x_rep.reshape(X_num, -1)
    X = torch.tensor(X)
    #X = test_x_rep
    y = test_y

    #digits_proj = TSNE(random_state=RS).fit_transform(X)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    X_embedded = TSNE(n_components=2).fit_transform(X)
    colours = ListedColormap(['#FB2A27', '#0057e7', '#FFA700', '#8134af', '#8bc24c', '#08D9D6'])
    #colours = ListedColormap(['#FB2A27', '#0057e7', '#FFA700', '#8134af'])
    #colours = ListedColormap(['#FB2A27', '#0057e7', '#FFA700', '#8134af', '#8bc24c', '#08D9D6', '#8AE1FC', '#f9bcdd','#ff8a5c'])
    plt.figure(facecolor='w', edgecolor='k', dpi=120)
    plt.grid(color='lightgray', linestyle = '--')#设置网格属性
    plt.grid(alpha=1)
    # plt.xlim(-10, 10)
    # plt.ylim(-4, 16)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    scat = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, marker=".", cmap=colours, s=160, alpha=1)
    #plt.title("Epilepsy", fontsize=22)
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_color('black')  # 设置上‘脊梁’为黑色
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    num1 = 1
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(handles=scat.legend_elements()[0], labels=['0', '1', '2', '3', '4', '5'], prop=font1, bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    #plt.show()

    # def scatter(x, colors):
    #     # We choose a color palette with seaborn.
    #     #palette = np.array(sns.color_palette("hls", 4))
    #     palette = [[0.50588, 0.20392, 0.68627], [0.12157, 0.39216, 0.03922], [0.86667, 0.16471, 0.48235], [0.99608, 0.85490, 0.46667]]
    #     palette = np.array(palette)
    #
    #     # We create a scatter plot.
    #     f = plt.figure(figsize=(8, 8))
    #     ax = plt.subplot(aspect='equal')
    #     sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=150,
    #                     c=palette[colors.astype(np.int)])
    #
    #     plt.xlim(-25, 25)
    #     plt.ylim(-25, 25)
    #
    #     ax.axis('on')
    #     ax.axis('tight')
    #     #ax = plt.gca()
    #
    #
    #     # We add the labels for each digit.
    #     # txts = []
    #     # for i in range(10):
    #     #     # Position of each label.
    #     #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     #     txt.set_path_effects([
    #     #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #     #         PathEffects.Normal()])
    #     #     txts.append(txt)
    #     #
    #     # return f, ax, sc, txts
    #     return f, ax, sc
    #
    # scatter(digits_proj, y)
    #
    # #plt.legend(label)
    if sne ==1:
        plt.savefig('JapaneseVowels_TSNE_Before.svg', dpi=120)
    else:
        plt.savefig('JapaneseVowels_TSNE_AFTER.svg', dpi=120)
    plt.show()

def draw_tsne_3D(test_x_rep, test_y, sne):
    X = torch.from_numpy(test_x_rep)
    y = test_y
    X_num = X.size(0)
    X = X.reshape(X_num, -1)
    X = torch.tensor(X)
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding_2d(X_tsne[:, 0:2], y)
    #plot_embedding_3d(X_tsne, y, "t-SNE 3D (time %.2fs)" % (time() - t0))
    if sne==1:
       plt.savefig('TSNE_Before.svg', dpi=120)
    else:
       plt.savefig('TSNE_After.svg', dpi=120)


    plt.show()
# 将降维后的数据可视化,2维
def plot_embedding_2d(X, y, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.patch.set_facecolor("none")  # 设置 ax1 区域背景颜色
    #ax.patch.set_alpha(0.5)  # 设置 ax1 区域背景颜色透明度
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    plt.xticks([])
    plt.yticks([])
    # for i in range(X.shape[0]):
    #     ax.text(X[i, 0], X[i, 1],'.',
    #              color=plt.cm.Set1(y[i]),
    #              fontdict={'weight': 'bold', 'size': 40})
    for i in range(X.shape[0]):
        if y[i] == 0:
            ax.text(X[i, 0], X[i, 1], '.',
                     color='#515bd4',
                     fontdict={'weight': 'light', 'size': 60})
        elif y[i] == 1:
            ax.text(X[i, 0], X[i, 1], '.',
                    color='#8134af',
                    fontdict={'weight': 'light', 'size': 60})
        elif y[i] == 2:
            ax.text(X[i, 0], X[i, 1], '.',
                    color='#dd2a7b',
                    fontdict={'weight': 'light', 'size': 60})
        else:
            ax.text(X[i, 0], X[i, 1], '.',
                    color='#feda77',
                    fontdict={'weight': 'light', 'size': 60})
    if title is not None:
        plt.title(title)

#%%
#将降维后的数据可视化,3维
def plot_embedding_3d(X, y, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure(edgecolor='black')
    # axes：坐标轴
    ax = plt.axes(projection='3d', facecolor = 'none')
    plt.grid(True, color='black')

    # ax=fig.add_subplot(111,projection='3d')#画子图




    #ax.patch.set_facecolor("none")
    #ax.grid(color='none', linestyle='- ', linewidth=4, alpha=0.3)
    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # for i in range(X.shape[0]):
    #     ax.text(X[i, 0], X[i, 1], X[i,2],'.',
    #              color=plt.cm.Set1(y[i]),
    #              fontdict={'weight': 'light', 'size': 40})
    for i in range(X.shape[0]):
        if y[i] == 0:
            ax.text(X[i, 0], X[i, 1], X[i,2],'.',
                     color='#515bd4',
                     fontdict={'weight': 'light', 'size': 40})
        elif y[i] == 1:
            ax.text(X[i, 0], X[i, 1], X[i, 2], '.',
                    color='#8134af',
                    fontdict={'weight': 'light', 'size': 40})
        elif y[i] == 2:
            ax.text(X[i, 0], X[i, 1], X[i, 2], '.',
                    color='#dd2a7b',
                    fontdict={'weight': 'light', 'size': 40})
        else:
            ax.text(X[i, 0], X[i, 1], X[i, 2], '.',
                    color='#feda77',
                    fontdict={'weight': 'light', 'size': 40})

    # plt.gca().spines['bottom'].set_color('black')  # x轴（spines脊柱）颜色设置
    # plt.gca().spines['bottom'].set_linewidth(10)  # x轴的粗细，下图大黑玩意儿就是这里的杰作
    # plt.gca().spines['bottom'].set_linestyle('--')  # x轴的线性
    if title is not None:
        plt.title(title)



