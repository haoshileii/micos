import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from . import drawsvm as draw
from models.lossestriples import hierarchical_contrastive_loss
# import matplotlib
# matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as ply
def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    #画表示前的测试实例tsne图
    #draw.draw_tsne_2D(test_data, test_labels, 1)
    train_repr = model.encode(train_data, train_labels, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, test_labels, encoding_window='full_series' if train_labels.ndim == 1 else None)
    #画原始数据的图形
    # print(test_labels[140])
    # plot_y = test_data[140].transpose(1, 0)
    # plot_x = list(range(1, test_data.shape[1] + 1))
    # fig, ax = ply.subplots()
    # for i in range(len(plot_y)):
    #     ax.plot(np.array(plot_x), plot_y[i] + 5 * i)
    #
    # # ply.legend()
    # ax.tick_params(bottom=False, top=False, left=False, right=False)  # 移除全部刻度线
    # ply.xticks([])
    # ply.yticks([])
    # ply.savefig('results/EplisyC4.svg', format='svg')


    #画表示后的图形
    #测试模型是否过拟合
    # loss_train_repr = hierarchical_contrastive_loss(
    #     1,
    #     train_repr,
    #     train_labels,
    #     temporal_unit=5
    # )
    # loss_test_repr = hierarchical_contrastive_loss(
    #     1,
    #     test_repr,
    #     test_labels,
    #     temporal_unit=5
    # )
    # print('/////////////////////////////////')
    # print(loss_train_repr)
    # print(loss_test_repr)
    # print('/////////////////////////////////')

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    #下面的if没有执行
    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)
    #是否SVM可视化
    #draw.draw_svm(test_repr, test_labels)
    #是否画T-SNE图
    #draw.draw_tsne_2D(test_repr, test_labels, 2)
    #是否画T-SNE图

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc }
