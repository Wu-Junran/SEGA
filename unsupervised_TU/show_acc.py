# -*- coding: utf-8 -*-


def show_acc(dataset):
    with open('logs/%s.out' % dataset) as fp:
        accs = [l.split() for l in fp if l.find('all') >= 0]
    g_accs = [accs[13:15] for accs in accs]
    t_accs = [accs[15:17] for accs in accs]
    h_accs = [accs[17:19] for accs in accs]
    accs = g_accs+t_accs+h_accs
    accs.sort(key=lambda ac: ac[0])
    print('Dataset:%-10s\tAcc:%s\tStd:%s' % (dataset, accs[-1][0], accs[-1][1]))


if __name__ == '__main__':
    datasets = ['NCI1', 'PROTEINS', 'DD', 'MUTAG', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY']
    for d in datasets:
        show_acc(d)
