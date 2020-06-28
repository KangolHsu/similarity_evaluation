from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score,f1_score
from prettytable import PrettyTable
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from PIL import Image
import math
import numpy as np
import io


# # Step 5: Get ROC Curves and TPR@FPR Table
def FAR_TAR():
    job='insightface'
    target='A200'
    files = [score_save_file]#人脸比对得到的相似度文件
    methods = []
    scores = []
    for file in files:
        methods.append(Path(file).stem)#模型方法名字
        scores.append(np.load(file))   #相似度

    methods = np.array(methods)
    scores = dict(zip(methods,scores))
    colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    #x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr) # select largest tpr at same fpr
        plt.plot(fpr, tpr, color=colours[method], lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc*100)))
        tpr_fpr_row = []
        tpr_fpr_row.append("%s-%s"%(method, target))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
            #tpr_fpr_row.append('%.4f' % tpr[min_index])
            tpr_fpr_row.append('%.2f' % (tpr[min_index]*100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10**-6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels) 
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True)) 
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB')
    plt.legend(loc="lower right")
    #plt.show()
    fig.savefig(os.path.join(save_path, '%s.pdf'%job))
    print(tpr_fpr_table)

def check_fit(truth, prob):
    """
    truth: 真实的值 [1,0,1,1,1]
    prob: 预测的值 [0.9,0.7,0.8,0.2,0.3]
    """
    fpr, tpr, threshold = roc_curve(truth, prob)     # drop_intermediate:(default=True) 
    roc_auc = auc(fpr, tpr)   # 计算auc值，roc曲线下面的面积 等价于 roc_auc_score(truth,prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.rcParams['savefig.dpi']=640
    plt.rcParams['figure.dpi']=640
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()

    buf = io.BytesIO()#
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return fpr, tpr,roc_auc,buf

def get_best_acc(scores,labels,thresholds):
    tprs = np.zeros(len(thresholds))
    fprs = np.zeros(len(thresholds))
    acc=np.zeros(len(thresholds))
    tp = tn = fp = fn = 0
    #far=[] false accept rate  == false positive rate 
    #tar=[] truth accrpt rate  == truth positive rate
    #for threshold in thresholds:
    for threshold_idx, threshold in enumerate(thresholds):
        tp = tn = fp = fn = 0      
        for index in range(0,len(scores)):
            dist = scores[index]
            actual_same=labels[index]
            predict_same  = dist > threshold
            #print(dist,actual_same,threshold)
            if predict_same and actual_same:
                tp += 1
            elif predict_same and not actual_same:
                fp += 1
            elif not predict_same and not actual_same:
                tn += 1
            elif not predict_same and actual_same:
                fn += 1
        #print(tp ,fp,tn ,fn)   
        if tp + fn == 0:
            tpr = 0
        else:
            tpr = float(tp) / float(tp + fn)#tar
        tprs[threshold_idx]=tpr

        if fp + tn == 0:
            fpr = 0
        else:
            fpr = float(fp) / float(fp + tn)#far
        fprs[threshold_idx]=fpr

        acc[threshold_idx]=float(tp + tn) / len(scores)
        maxindex  = np.argmax(acc)
        best_accuracy=acc[maxindex]
        best_threshold=thresholds[maxindex]
    return best_accuracy,best_threshold


def main():
    # score_files=['mobilefacenet_cos_sim_A200_0_0.txt',
                 # 'mobilefacenet_cos_sim_A200_0_30.txt',
                 # 'mobilefacenet_cos_sim_A200_30_30.txt',
                 # 'mobilefacenet_cos_sim_IPhone_0_0.txt',
                 # 'mobilefacenet_cos_sim_IPhone_0_30.txt',
                 # 'mobilefacenet_cos_sim_IPhone_30_30.txt'
                 # ]
    data_types=['A200_0_0',
                 'A200_0_30',
                 'A200_30_30',
                 'IPhone_0_0',
                 'IPhone_0_30',
                 'IPhone_30_30'
                 ]
    score_files_dict = {'A200_0_0': 'mobilefacenet_cos_sim_A200_0_0.txt',
                        'A200_0_30': 'mobilefacenet_cos_sim_A200_0_30.txt', 
                        'A200_30_30': 'mobilefacenet_cos_sim_A200_30_30.txt',
                        'IPhone_0_0': 'mobilefacenet_cos_sim_IPhone_0_0.txt',
                        'IPhone_0_30': 'mobilefacenet_cos_sim_IPhone_0_30.txt',
                        'IPhone_30_30': 'mobilefacenet_cos_sim_IPhone_30_30.txt'
                        }
    plt.figure()
    fprs=[]
    tprs=[]
    x_labels = [10**-4,10**-3, 10**-2, 10**-1]
    #x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
    tpr_fpr_table = PrettyTable(['score_files'] + [str(x) for x in x_labels] + ['AUC','bestAcc','bestThr','F1_score','RMSE'])
    for data_type in data_types:
        print(score_files_dict[data_type])
        score_labels=open(score_files_dict[data_type],'r')
        scores=[]#similarity
        labels=[]#[0,0,0,,1,1,1,......]
        for data in score_labels:
            splits=data.replace('\n','').split(' ')
            #print(splits)
            scores.append(float(splits[0]))
            labels.append(int(splits[1]))
        #fpr,tpr,auc_value,buf=check_fit(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)     # drop_intermediate:(default=True) 
        auc_value = auc(fpr, tpr)   # 计算auc值，roc曲线下面的面积 等价于 roc_auc_score(truth,prob)
        plt.plot(fpr, tpr, lw=2, label='%s (auc = %0.6f)' % (data_type,auc_value))


        fprs.append(fpr)
        tprs.append(tpr)
        tpr_fpr_row = []
        tpr_fpr_row.append(score_files_dict[data_type])
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
            #print('TAR@FAR=%f : %.6f' % (x_labels[fpr_iter],tpr[min_index]))
            tpr_fpr_row.append('%.6f' % (tpr[min_index]*100))
        tpr_fpr_row.append('%.6f' % (auc_value))
        #roc_curve = Image.open(buf)
        #roc_curve.show()
        #ROC_path=file+'.figure.png'
        #roc_curve.save(ROC_path, quality=95, subsampling=0)
        #print('AUC Value : %.6f ' % (auc_value))

        thresholds = np.arange(0, 1, 0.0001)#
        best_acc=0
        best_thr=0
        for threshold in thresholds:
            predics = [1 if i>=threshold else 0 for i in scores]
            acc=accuracy_score(labels, predics)
            if acc>best_acc:
                best_acc=acc
                best_thr=threshold
        #print('Best Acc : %.6f ,thr : %.6f' % (best_acc,best_thr))
        tpr_fpr_row.append('%.6f' % (best_acc))
        tpr_fpr_row.append('%.6f' % (best_thr))
        preds = [1 if i>=best_thr else 0 for i in scores]
        F1=f1_score(labels, preds, labels=None, pos_label=1, average='binary', sample_weight=None)# F1值
        #F1 = 2 * (precision * recall) / (precision + recall) precision(查准率)=TP/(TP+FP) recall(查全率)=TP/(TP+FN)
        #print("F1_score :%.6f"%(F1))
        tpr_fpr_row.append('%.6f' % (F1))

        #best_accuracy,best_threshold=get_best_acc(scores,labels,thresholds)
        #print('Best Acc : %.4f ,thr : %.4f' % (best_accuracy,best_threshold))

        tpr_fpr_row.append('%.6f' % (math.sqrt(mean_squared_error(labels, preds))))
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.rcParams['savefig.dpi']=640
    plt.rcParams['figure.dpi']=640
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig('plot.png',dpi=640,bbox_inches = 'tight')
    print(tpr_fpr_table)

if __name__ == '__main__':
    main()
    # print(__name__)