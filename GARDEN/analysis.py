import shap
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder


def shap_DEG_with_reconstruction(adata):
    X = adata.obsm['emb']
    y = adata.obs['domain']

    # 将标签编码为数字
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 训练模型
    model = RandomForestClassifier()
    model.fit(X, y_encoded)

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 获取不同标签的SHAP值
    for i, class_name in enumerate(label_encoder.classes_):
        print(f'Class: {class_name}')
        plt.rcParams["figure.figsize"] = (3, 3)
        # 绘制SHAP值的可视化
        shap.summary_plot(shap_values[i], X, feature_names=adata.var_names, show=False)
        plt.title(f'SHAP Values for {class_name}')
        plt.show()

def detect_meta_genes_ez_mode(adata, target,domain_name,start_gene, use_raw=False):
    meta_name, meta_exp=find_meta_gene(input_adata=adata,
                    pred=adata.obs[domain_name].tolist(),
                    target_domain=target,
                    start_gene=start_gene,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=3)
    print("Meta gene:", meta_name)
    return meta_exp
def find_meta_gene(input_adata,
                    pred,
                    target_domain,
                    start_gene,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=5,
                    use_raw=False):
    #初始选取的 
    meta_name=start_gene
    #copy
    adata=input_adata.copy()
    adata.X = adata.X.toarray()   
    #初始化meta列
    adata.obs["meta"]=adata.X[:,adata.var.index==start_gene]#.getnnz()
    adata.obs["pred"]=pred
    #所有细胞或空间点的总数
    num_non_target=adata.shape[0]
    #逐步优化元基因
    for i in range(max_iter):
        #Select cells
        #从adata中选择meta基因表达值高于目标域平均值，或属于目标域的空间点，存储在tmp（adata对象）中
        tmp=adata[((adata.obs["meta"]>np.mean(adata.obs[adata.obs["pred"]==target_domain]["meta"]))|(adata.obs["pred"]==target_domain))]
        # 在 tmp 中添加一个名为 target 的列，将属于目标域的标记为 1，其他的标记为 0。
        tmp.obs["target"]=((tmp.obs["pred"]==target_domain)*1).astype('category').copy()
        #如果 target 中的类别少于两个，或者其中一个类别的数量少于 5，则打印当前的 meta_name 并返回元基因名及其表达值列表
        if (len(set(tmp.obs["target"]))<2) or (np.min(tmp.obs["target"].value_counts().values)<5):
            print("Meta gene is: ", meta_name)
            return meta_name, adata.obs["meta"].tolist()

        #DE
        sc.tl.rank_genes_groups(tmp, groupby="target",reference="rest", n_genes=1,method='wilcoxon')
        adj_g=tmp.uns['rank_genes_groups']["names"][0][0]
        add_g=tmp.uns['rank_genes_groups']["names"][0][1]
        # adj_g,add_g = calculate_SHAP_gene(tmp)
        meta_name_cur=meta_name+"+"+add_g+"-"+adj_g
        print("Add gene: ", add_g)
        print("Minus gene: ", adj_g)
        
        #Meta gene
        adata.obs[add_g]=adata.X[:,adata.var.index==add_g]
        adata.obs[adj_g]=adata.X[:,adata.var.index==adj_g]
        adata.obs["meta_cur"]=(adata.obs["meta"]+adata.obs[add_g]-adata.obs[adj_g])
        adata.obs["meta_cur"]=adata.obs["meta_cur"]-np.min(adata.obs["meta_cur"])
        mean_diff_cur=np.mean(adata.obs["meta_cur"][adata.obs["pred"]==target_domain])-np.mean(adata.obs["meta_cur"][adata.obs["pred"]!=target_domain])
        num_non_target_cur=np.sum(tmp.obs["target"]==0)
        if (early_stop==False) | ((num_non_target>=num_non_target_cur) & (mean_diff<=mean_diff_cur)):
            num_non_target=num_non_target_cur
            mean_diff=mean_diff_cur
            print("Absolute mean change:", mean_diff)
            print("Number of non-target spots reduced to:",num_non_target)
        else:
            print("Stopped!", "Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
            print("Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
            print("Previous Number of non-target spots",num_non_target)
            print("Current Number of non-target spots",num_non_target_cur)
            print("Absolute mean change", mean_diff)
            print("===========================================================================")
            print("Meta gene: ", meta_name)
            print("===========================================================================")
            return meta_name, adata.obs["meta"].tolist()
        meta_name=meta_name_cur
        adata.obs["meta"]=adata.obs["meta_cur"]
        print("===========================================================================")
        print("Meta gene is: ", meta_name)
        print("===========================================================================")
    return meta_name, adata.obs["meta"].tolist()


def calculate_SHAP_gene(adata,embedding='emb',label = 'domain'):
    # 假设你已经有了 X 和 y
    X = adata.obsm[embedding]
    y = adata.obs[label]

    # 将标签编码为数字
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 训练模型
    model = RandomForestClassifier()
    model.fit(X, y_encoded)

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 对每个类分别计算SHAP值
    for i, class_name in enumerate(label_encoder.classes_):
        print(f'Class: {class_name}')
        plt.rcParams["figure.figsize"] = (3, 3)
        shap.summary_plot(shap_values[i], X, feature_names=adata.var_names, show=False)
        plt.title(f'SHAP Values for {class_name}')
        plt.show()

    # 获取目标域对应的 SHAP 值
    target_shap_values = shap_values[np.where(label_encoder.classes_ == target_domain)[0][0]]

    # 计算 SHAP 值的均值
    shap_mean = np.mean(target_shap_values, axis=0)

    # 按照SHAP值排序，找到对目标域影响最大的基因
    shap_sorted_idx = np.argsort(shap_mean)

    # 选择最重要的正向和负向基因
    adj_g = adata.var_names[shap_sorted_idx[0]]  # SHAP值最小的基因
    add_g = adata.var_names[shap_sorted_idx[-1]] # SHAP值最大的基因

    return adj_g,add_g


