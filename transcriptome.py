# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:21:10 2019

@author: liushang
"""
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
print('>===import the basic parcels successfully===<')
print('>===the package needed are correctly imported===<')
parser=argparse.ArgumentParser(description=
                               'this is the pipeline for transcriptome data analysis')
parser.add_argument('-df','--dataframe',type=str,help='the path of transcriptome expression file')
parser.add_argument('-rf','--result',type=str,help='the path of result directory')
parser.add_argument('-f','--features',type=str,help='features you want to analyze',nargs='*')
parser.add_argument('-c','--color',type=str,help='colors or a file contain the colors end with txt',nargs='*')
parser.add_argument('-thr','--threshold',type=float,help='the threshold we need in different analysis',nargs='*')
parser.add_argument('-p','--plot',type=str,help='the type you choose to visualize data',
                    choices=['hist','density','box_plot','scatter','distr_2d','pie','vocano_plot','cor_plot','kmeans','pca','hierarchy'])
args=parser.parse_args()
#读入数据表
if args.dataframe[-3:]=='txt':
    dataframe=pd.read_csv(args.dataframe,sep='\t')
    print('the input dataframe file has been correctly imported')
elif args.dataframe[-3:]=='csv':
    dataframe=pd.read_csv(args.dataframe,sep=',')
else:
    print('the format of the input dataframe file is incorrect')
#指定输出文件
result_dic=args.result
print('the result directory is assigned,it\'s %s'%args.result)
features=args.features
print('the features have been storaged')
if args.plot=='hist':
    print('starting hist plot')
    def hist(dataframe,features,color,result):
        if os.path.isdir(result):
            pass
        else:
            os.mkdir(result)
        for i,j in zip(features,color):
            n,bins,patches=plt.hist(dataframe[i],color=j,edgecolor='k',bins=10,rwidth=0.9)
            sigma,mean=dataframe[i].std(),dataframe[i].mean()                       
            plt.xlabel(i)
            plt.ylabel('number')
            plt.title('the distibution of %s (mean=%.3f,std=%.3f)'%(i,mean,sigma))
            plt.savefig(result+'/hist%s.png'%i)
            plt.clf()
    if len(args.color)!=len(args.features):
        print('the number of features is not compatible with number of colors')
    hist(dataframe,args.features,args.color,args.result+'/hist')
    print('hist plot is finished')
elif args.plot=='density':    
    def density(dataframe,features,color,result):
            if os.path.isdir(result):
                pass
            else:
                os.mkdir(result)
            for i,j in zip(features,color):
                dataframe[i].plot(kind='kde',color=j)
                sigma,mean=dataframe[i].std(),dataframe[i].mean()
                plt.xlabel(i)
                plt.title('the distibution of %s (mean=%.3f,std=%.3f)'%(i,mean,sigma))
                plt.savefig(result+'/density%s.png'%i)
                plt.clf()
    if len(args.color)!=len(args.features):
        print('the number of features is not compatible with number of colors')
    density(dataframe,args.features,args.color,args.result+'/density')
#箱线图
elif args.plot=='box_plot':
    print('starting the boxplot')    
    def box_plot(dataframe,features,colors,result):
        if os.path.isdir(result):
                pass
        else:
                os.mkdir(result)
        plot_list=[]
        for i in features:
            plot_list.append(dataframe[i])
        box_list=plt.boxplot(plot_list,labels=features,showmeans=True,sym='.',patch_artist=True,
                             flierprops={'marker':'o','markerfacecolor':'black'})
        for box,color in zip(box_list['boxes'],colors):
            box.set_facecolor(color)
        plt.savefig(result+'/boxplot.png')
        plt.clf()
    if len(args.color)!=len(args.features):
        print('the number of features is not compatible with number of colors')
    box_plot(dataframe,args.features,args.color,args.result+'/box_plot')
    print('the boxplot was finished')
#散点图
elif args.plot=='scatter':
    print('starting scatter plot')
    def scatter(dataframe,features,result,color='r'):
             if os.path.isdir(result):
                pass
             else:
                os.mkdir(result)
             if len(features)<2:
                print('lack %d feature in the scatter'%(2-len(features)))
                return None
             elif len(features)>2:
                print('two many features in the scatter,only need two!')
                return None
             if len(color)>1:
                print('the color are two many,only one is enough!')
             elif len(color)==0:
                print('you didn\'t assign the color, we use red as default')
             plt.scatter(dataframe[features[0]],dataframe[features[1]],c=color)
             plt.xlabel(features[0])
             plt.ylabel(features[1])
             plt.title('the distribution of %s & %s'%(features[0],features[1]))
             plt.savefig(result+'/scatter.png')
             plt.clf()
    scatter(dataframe,args.features,args.result+'/scatter',color='b')
    print('scatter plot has been finished')
#二维分布图
elif args.plot=='distr_2d':    
    def distr_2d(dataframe,features,result):
        if os.path.isdir(result):
                pass
        else:
                os.mkdir(result)
        if len(features)<2:
                print('lack %d feature in the scatter'%(2-len(features)))
                return None
        elif len(features)>2:
                print('two many features in the scatter,only need two!')
                return None
        plt.hist2d(dataframe[features[0]],dataframe[features[1]],color='black',cmap='Reds')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.savefig(result+'/%s_%sdistr_2d.png'%(features[0],features[1]))
        plt.clf()
    distr_2d(dataframe,args.features,args.result+'/distr_2d')
#饼图
elif args.plot=='pie':
    if len(args.threshold)>1:
        print('only need one threshold')
    def pie(dataframe,features,result,threshold=1):
        if os.path.isdir(result):
                pass
        else:
                os.mkdir(result)
        for i in features:
            dic={}
            sum_un=0
            sum_ex=0
            for j in iter(dataframe[i]<threshold):
                if j==True:
                    sum_un+=1
                else:
                    sum_ex+=1
            dic['ex']=sum_ex/(sum_ex+sum_un)
            dic['un']=sum_un/(sum_ex+sum_un)
            data=pd.Series(dic)
            data.name=''
            plt.axes(aspect='equal')
            data.plot(kind='pie',
                      autopct='%.1f%%',
                      radius=1,
                      startangle=180,
                      counterclock=False,
                      title='distribution of %s'%i)
            plt.savefig(result+'/%s_pie.png'%i)
            plt.clf()
    pie(dataframe,args.features,args.result+'/pie',args.threshold[0])
#火山图
elif args.plot=='vocano_plot':
    if len(args.threshold)!=2:
        print('we need two threshold: the names of pvalue and foldchange')    
    def vocano_plot(dataframe,result,colors,pvalue,foldchange):
        dataframe['treated_p']=-np.log10(dataframe[pvalue])
        dataframe['class']='normal'
        dataframe.loc[(dataframe[foldchange]>1)&(dataframe[pvalue]<0.05),'class']='up'
        dataframe.loc[(dataframe[foldchange]<-1)&(dataframe[pvalue]<0.05),'class']='down'
        ax=sns.scatterplot(x=foldchange,y='treated_p',
                           hue='class',
                           hue_order=('down','normal','up'),
                           palette=(colors[0],'grey',colors[1]),
                           data=dataframe)
        ax.set_ylabel('-log10(p)',fontweight='bold')
        ax.set_xlabel('foldchange',fontweight='bold')  
        plt.savefig(result+'/vocano.png')
    vocano_plot(dataframe,args.result+'/vocano',args.color,args.threshold[0],args.threshold[1])
#火山图后必须执行这一个
    dataframe=dataframe[dataframe.columns.tolist()[:-2]]    
#相关热图
elif args.plot=='cor_plot':
    def cor_plot(dataframe,features,result):
        if os.path.isdir(result):
                pass
        else:
                os.mkdir(result)
        df_cor=dataframe[features].corr()
        plt.figure(figsize=(8,6),dpi=80)
        sns.heatmap(df_cor,xticklabels=df_cor.columns,yticklabels=df_cor.columns,cmap='RdYlBu_r',linewidth=0.1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(result+'/cor_plot.png')
        plt.clf()
    cor_plot(dataframe,args.features,args.result+'/cor_plot')
#均值
elif args.plot=='kmeans':   
    def kmeans(df,features,result):
        column_names=df.columns.tolist()
        import sklearn
        from sklearn.cluster import KMeans
        print('import Kmeans correctly')
        df_kmeans=df[features]
        k_class=[]
        for i in range(2,len(df_kmeans)):
            k_clust=KMeans(n_clusters=i)
            k_labels=k_clust.fit_predict(df_kmeans)
            k_class.append((i,sklearn.metrics.silhouette_score(df_kmeans,k_labels)))
        proper_k=0
        k_score=0
        for i in k_class:
            if i[1]>k_score:
                k_score=i[1]
                proper_k=i[0]
            else:
                continue
        kmeans_cluster=KMeans(n_clusters=proper_k).fit(df_kmeans)
        df_kmeans['clusters']=kmeans_cluster.labels_
        df_kmeans[column_names[0]]=df[column_names[0]]
        df_kmeans.to_csv(result+'/kmeans.csv',index=False)
        del df_kmeans
    kmeans(dataframe,args.features,args.result+'/kmeans')
#pca
elif args.plot=='pca':
    if len(args.threshold)!=1:
        print('you must input a number to assign the number of components')
    def pca(df,nun_comp,result):
        nun_comp=int(nun_comp)
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D
        print('import PCA and 3d_ploting successfully')
        pca_single=PCA(n_components=nun_comp)
        pca_result=pca_single.fit_transform(df[df.columns.tolist()[1:]])
        if nun_comp>3:
            pca_result.to_csv('pcs_result.csv',index=False)
        elif nun_comp==3:
            plt.figure(figsize=(15,10))
            ax=plt.subplot(projection='3d')
            pca_result_df=pd.DataFrame()
            pca_result_df['pca1']=pca_result[:,0]
            pca_result_df['pca2']=pca_result[:,1]
            pca_result_df['pca3']=pca_result[:,2]
            ax.scatter(pca_result_df.iloc[:,0],
                           pca_result_df.iloc[:,1],
                           pca_result_df.iloc[:,2])
            ax.set_xlabel('PCA1')
            ax.set_ylabel('PCA2')
            ax.set_zlabel('PCA3')
            plt.title('explain ratio:%.3f, %.3f, %.3f'%(pca_single.explained_variance_ratio_[0],
                                                        pca_single.explained_variance_ratio_[1],
                                                        pca_single.explained_variance_ratio_[2]))
            plt.savefig(result+'/pca3d.png')
            plt.clf()
        elif nun_comp==2:
            plt.figure(figsize=(10,8))          
            plt.scatter(pca_result[:,0],
                        pca_result[:,1])
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.title('explain ratio:%.3f, %.3f'%(pca_single.explained_variance_ratio_[0],
                                                pca_single.explained_variance_ratio_[1]))
            plt.savefig(result+'/pca2d.png')
            plt.clf()
    pca(dataframe,args.threshold[0],args.result+'/pca') 
elif args.plot=='hierarchy':  
    def hierarchy(df,features,result): 
        from scipy.cluster import hierarchy
        print('import the hierarchy successfully')
        df_hierarchy=df
        df_hierarchy=df_hierarchy.set_index(df.columns.tolist()[0])
        plt.figure(figsize=(15,10))
        sns.clustermap(df_hierarchy[features],method ='ward',metric='euclidean',cmap='RdYlBu_r',linewidths=0.5)
        plt.savefig(result+'/hierarchy.png')
        plt.clf()
    hierarchy(dataframe,args.features,args.result+'/hierarchy')
print('>===the analysis was finished===<')
print('***Thank you for using transcriptome visualizing***')