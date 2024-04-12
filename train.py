import torch
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
import sklearn
import numpy as np
from option import args
from model.tgat import TGAT
from utils import EarlyStopMonitor, logger_config, LoadBatchData
from tqdm import tqdm
import datetime, os
import warnings

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)




def criterion(prediction_dict, labels, model, config, device):

    for key, value in prediction_dict.items():
        if key != 'root_embedding' and key != 'group' and key != 'dev' and key != 'cls':
            prediction_dict[key] = value[labels > -1]

    labels = labels[labels > -1]
    labels = labels.clone().detach()
    # labels2 = labels.clone()
    labels = torch.tensor(labels, dtype=float)
    logits = prediction_dict['logits']
    
    loss_classify = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction='none')
    loss_classify = torch.mean(loss_classify)
    loss2 = ce(logits.double(), labels.float())
    # loss3 = ce2(prediction_dict['cls'].double().softmax(1), torch.cuda.LongTensor(labels2).squeeze())
    # loss2 = loss2 / logits.shape[0]
    # loss3 = 
    loss = 0.5 * loss_classify.clone() + loss2.clone()
    loss_anomaly = torch.Tensor(0).to(device)
    loss_supc = torch.Tensor(0).to(device)
    alpha = config.anomaly_alpha  # 1e-1
    beta = config.supc_alpha # 1e-3
    if config.mode == 'sad':
        loss_anomaly = model.gdn.dev_loss(torch.squeeze(labels), torch.squeeze(prediction_dict['anom_score']), torch.squeeze(prediction_dict['time']))
        loss_supc = model.suploss(prediction_dict['root_embedding'], prediction_dict['group'], prediction_dict['dev'])
        loss += alpha * loss_anomaly + beta * loss_supc
        # loss += alpha * loss_anomaly

    return loss, loss2, loss_anomaly, loss_supc

def eval_epoch(dataset, model, config, device):
    loss = 0
    m_loss, m_pred, m_label = [], [], []
    m_dev = []
    probs = None
    with torch.no_grad():
        model.eval()
        for batch_sample in dataset:
            x = model(
                batch_sample['src_edge_feat'].to(device),
                batch_sample['src_edge_to_time'].to(device),
                batch_sample['src_center_node_idx'].to(device),
                batch_sample['src_neigh_edge'].to(device),
                batch_sample['src_node_features'].to(device),
                batch_sample['current_time'].to(device),
                batch_sample['src_center_node_idx_ori'].to(device),
                batch_sample['src_neigh_edge_ori'].to(device),
                batch_sample['labels'].to(device)
            )
            y = batch_sample['labels'].to(device)
            dev_score = x['dev'].cpu().numpy().flatten()
            m_loss = np.concatenate((m_loss, criterion(x, y, model, config, device)[1].cpu().numpy().flatten()))

            # pred_score = x['logits'].sigmoid().cpu().numpy().flatten()
            y = y.cpu().numpy().flatten()
            
            m_label = np.concatenate((m_label, y))
            m_dev = np.concatenate((m_dev, dev_score))
            cls = x['cls']
            prob = cls.softmax(1)
            if probs is None:
                probs = prob.cpu().numpy()
            else:
                probs = np.concatenate((probs, prob.cpu().numpy()))
            

            # m_pred = np.concatenate((m_pred, pred_score))
    best_thre = get_best_f1(m_label, probs)
    m_pred = np.zeros(probs.shape[0])
    m_pred[probs[:, 1] > best_thre] = 1
    auc_roc = sklearn.metrics.roc_auc_score(m_label, probs[:, 1])
    # auc_roc = sklearn.metrics.roc_auc_score(m_label, m_pred)
    # precision, recall, thresholds = sklearn.metrics.precision_recall_curve(m_label, m_pred, pos_label=1)
    # m_pred[m_pred < 0.5] = 0
    # m_pred[m_pred > 0.5] = 1
    # acc = sklearn.metrics.accuracy_score(m_label, m_pred)
    # pr_auc = sklearn.metrics.auc(recall, precision)
    
    pr_auc = sklearn.metrics.average_precision_score(m_label, m_pred, average='macro', pos_label=1, sample_weight=None)
    # print('max:{},  min:{}'.format(np.max(probs[:, 1]), np.min(probs[:, 1])))
    # print(sklearn.metrics.classification_report(m_label, m_pred, digits=4))
    conf = sklearn.metrics.confusion_matrix(m_label, m_pred)
    g_mean = gmean(conf)
    return auc_roc, np.mean(m_loss), m_dev, m_label, pr_auc, g_mean

def gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.01, 0.25, 25):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = sklearn.metrics.f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_thre


# lbd = LoadBatchData(config, 'test')




if __name__ == '__main__':
    with torch.cuda.device(0):
        ce = torch.nn.CrossEntropyLoss()
        # ce2 = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.05, 0.95])))
        set_random_seed(1024)
        config = args
        if config.data_set == 'zhuanzhang':
            config.type_num = 0
            config.range = []
        else:
            config.type_num = 1
            config.range = [13236]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ce2 = ce2.to(device)
        # device = torch.device("cpu")
        # log file name set
        now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_base_path = f"{os.getcwd()}/train_log"
        file_list = os.listdir(log_base_path)
        max_num = [0] # [int(fl.split("_")[0]) for fl in file_list if len(fl.split("_"))>2] + [-1]
        log_base_path = f"{log_base_path}/{max(max_num)+1}_{now_time}"
        # log and path
        get_checkpoint_path = lambda epoch: f'{log_base_path}saved_checkpoints/{args.data_set}-{args.mode}-{args.module_type}-{args.mask_ratio}-{epoch}.pth'
        logger = logger_config(log_path=f'{log_base_path}/log.txt', logging_name='gdn')
        logger.info(config)

        dataset_train = dataset.DygDataset(config, 'train')
        dataset_valid = dataset.DygDataset(config, 'valid')
        dataset_test = dataset.DygDataset(config, 'test')
        config.node_num = dataset_train.node_features.shape[0]
        all_node_features = torch.from_numpy(dataset_train.node_features)
        gpus = None if config.gpus == 0 else config.gpus

        # node feature
        collate_fn = dataset.Collate(config)

        backbone = TGAT(config, device)
        model = backbone.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=config.batch_size,
            shuffle=False,
            #shuffle=True,
            # num_workers=config.num_data_workers,
            pin_memory=True,
            #sampler=dataset.RandomDropSampler(dataset_train, 0.75),   #for reddit
            collate_fn=collate_fn.dyg_collate_fn
        )

        loader_valid = torch.utils.data.DataLoader(
            dataset=dataset_valid,
            batch_size=config.batch_size,
            shuffle=False,
            #shuffle=True,
            # num_workers=config.num_data_workers,
            collate_fn=collate_fn.dyg_collate_fn
        )


        loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=config.batch_size,
            shuffle=False,
            #shuffle=True,
            # num_workers=config.num_data_workers,
            collate_fn=collate_fn.dyg_collate_fn
        )

        max_val_auc, max_test_auc = 0.0, 0.0
        early_stopper = EarlyStopMonitor()
        best_auc = [0, 0, 0]
        num = 0
        # for batch_sample in loader_train:
        #     num += 1
        #     print(num)


        for epoch in range(config.n_epochs):
            ave_loss = 0
            count_flag = 0
            m_loss, auc = [], []
            loss_anomaly_list = []
            loss_class_list = []
            loss_supc_list = []
            dev_score_list = np.array([])
            dev_label_list = np.array([])
            with tqdm(total=len(loader_train)) as t:
                for batch_sample in loader_train:
                    count_flag += 1
                    t.set_description('Epoch %i' % epoch)
                    optimizer.zero_grad()
                    model.train()
                    x = model(
                        batch_sample['src_edge_feat'].to(device),
                        batch_sample['src_edge_to_time'].to(device),
                        batch_sample['src_center_node_idx'].to(device),
                        batch_sample['src_neigh_edge'].to(device),
                        batch_sample['src_node_features'].to(device),
                        batch_sample['current_time'].to(device),
                        batch_sample['src_center_node_idx_ori'].to(device),
                        batch_sample['src_neigh_edge_ori'].to(device),
                        batch_sample['labels'].to(device)
                    )
                    y = batch_sample['labels'].to(device)
                    dev_score = x["dev"]
                    loss, loss_classify, loss_anomaly, loss_supc = criterion(x, y, model, config, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
                    optimizer.step()

                    # get training results
                    with torch.no_grad():
                        model = model.eval()
                        m_loss.append(loss.item())
                        pred_score = x['logits'].sigmoid()

                        dev_score_list = np.concatenate((dev_score_list, dev_score.cpu().numpy().flatten()))
                        dev_label_list = np.concatenate((dev_label_list, batch_sample['labels'].cpu().numpy().flatten()))


                    loss_class_list.append(loss_classify.detach().clone().cpu().numpy().flatten())
                    if config.mode == 'gdn':
                        loss_anomaly_list.append(loss_anomaly.detach().clone().cpu().numpy().flatten())
                        t.set_postfix(loss=np.mean(loss_class_list), loss_anomaly=np.mean(loss_anomaly_list))
                    elif config.mode == 'sad':
                        loss_anomaly_list.append(loss_anomaly.detach().clone().cpu().numpy().flatten())
                        loss_supc_list.append(loss_supc.detach().clone().cpu().numpy().flatten())
                        t.set_postfix(loss=np.mean(loss_class_list), loss_anomaly=np.mean(loss_anomaly_list), loss_sup=np.mean(loss_supc_list))
                    else:
                        t.set_postfix(loss=np.mean(loss_class_list))
                    t.update(1)


            # val_auc, val_loss, val_m_dev, val_m_label, val_pr_auc, val_acc = eval_epoch(loader_valid, model, config, device)
            test_auc, test_loss, test_m_dev, test_m_label, test_pr_auc, g_mean = eval_epoch(loader_test, model, config, device)
            max_test_auc = max(max_test_auc,test_auc)
            if test_auc>best_auc[1]:
                best_auc = [epoch, test_auc]
                torch.save(model.state_dict(), './trained_model/' + config.data_set + '_model_' + config.model_type + '_' + str(config.train_ratio) + '.pth')
                


            logger.info('\n epoch: {}'.format(epoch))
            logger.info(f'train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, anomaly loss: {np.mean(loss_anomaly_list)}, sup loss: {np.mean(loss_supc_list)}')
            # logger.info('val mean loss:{}, val auc:{}'.format(val_loss, val_auc))
            logger.info('test mean loss:{}, test auc:{}, test pr_auc:{}, test gmean:{}'.format(test_loss, test_auc, test_pr_auc, g_mean))
            # logger.info('val pr_auc:{}, test pr_auc:{}'.format(val_pr_auc, test_pr_auc))
            # logger.info('val acc:{}, test acc:{}'.format(val_acc, test_acc))



            # if early_stopper.early_stop_check(val_auc):
            #     logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            #     #print(f'Loading the best model at epoch {early_stopper.best_epoch}')
            #     #best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            #     #model.load_state_dict(torch.load(best_model_path))
            #     #print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            #     #model.eval()
            #     break
            # else:
            #     #torch.save(model.state_dict(), get_checkpoint_path(epoch))
            #     pass

            #记录下score的结果
        #     dev_score_list = np.concatenate((dev_score_list, val_m_dev, test_m_dev))
        #     dev_label_list = np.concatenate((dev_label_list, val_m_label, test_m_label))
        #     output_file = './graph_dev_score.txt'
        #     with open(output_file, 'w') as fout:
        #         for i, (score, label) in enumerate(zip( dev_score_list, dev_label_list)):
        #             fout.write(f'{i}\t')
        #             fout.write(f'{score}\t')
        #             fout.write(f'{label}\n')
        #             pass
        #         pass
        #     pass

        logger.info(f'\n max_test_auc: {max_test_auc}')
        logger.info('\n best auc: epoch={},  test={}'.format(best_auc[0], best_auc[1]))
        # model.load_state_dict(torch.load('./trained_model/zhuanzhang_model_all.pth'))
        # print('xxxxxxxxxx')




