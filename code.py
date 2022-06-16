"""
Code to extract the informations used in the paper: 
'Time stability and Connectivity Analysis with an Intracortical 96-channel Microelectrode Array inserted in human visual cortex'
Authors: Fabrizio Grani, Cristina Soto Sanchez, Fernando Daniel Farfan, Arantxa Alfaro, Maria Dolores Grima, Alfonso Rodil Doblado, Eduardo Fernández Jover
"""

#%% import sections
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import copy
import scipy.stats as statistic
import math
import pywt
import sklearn
import random

#%% definition of functions
def coh_utah_map01(coherence_matrix,stim_el):
    plot_vector=np.ones([1,100])
    plot_vector[0,[0,9,90,99]+stim_el]=0
    fig=plt.figure()
    index_plot=0
    index_electrodes=0
    for i in range(10):
        for j in range(10):
            plt.subplot(10,10,index_plot+1)
            if plot_vector[0,index_plot]==1:
                coh_ch=coherence_matrix[index_electrodes,:]
                index_electrodes=index_electrodes+1
                coh_ch=np.insert(coh_ch,0,0)
                coh_ch=np.insert(coh_ch,9,0)
                for iii in range(len(stim_el)):
                    coh_ch=np.insert(coh_ch,stim_el[iii],0)    
                coh_ch=np.insert(coh_ch,90,0)
                coh_ch=np.insert(coh_ch,99,0)
                coh_ch=np.reshape(coh_ch,[10,10])
                plt.pcolormesh(coh_ch,vmin=0,vmax=1)
                ax = plt.gca()
                ax.invert_yaxis()
            plt.xticks([])
            plt.yticks([])
                
            index_plot=index_plot+1 
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    plt.colorbar(cax=cbar_ax)

#%% load the data
lfp = np.load('lfp_1KHz_30days_pre.npy')   #the data included are related to 30 days post implantation

#PARAMETERS 
fs=1000    #sampling frequency

#colors
cmap=plt.get_cmap('Blues')
line_colors = cmap(np.linspace(0,1,50))
c1=line_colors[40]
c2=c=line_colors[30]

#DISTANCE BETWEEN THE ELECTRODES
distance=np.zeros([100,100])
index1=-1
index2=-1
for i in range(10):
    for j in range(10):
        p1=[i,j]
        index1=index1+1
        index2=-1
        for ii in range(10):
            for jj in range(10):
                p2 = [ii,jj]
                index2=index2+1
                distance[index1,index2]=math.dist(p1,p2)
null = np.array([0,9,90,99])
distance = np.delete(distance,null,axis = 0)
distance = np.delete(distance,null,axis = 1)
distance = distance * 0.4   #distance in millimeters in the Utah array

#%% VARIANCE
variance=np.zeros([96,30])

for channel in range(96):
    var=[]
    for time in np.arange(0,60*fs,2*fs):
        var.append(np.var(lfp[channel,time:time+2*fs]))
    variance[channel,:]=var

mean_variance=np.mean(variance,1)   #mean variance of each channel
std_variance=np.std(variance,1)     #standard deviation of the variance of each channel

#%% POWER SPECTRAL DENSITY
PSD=[]
for channel in range(96):
    f, Pxx = signal.welch(lfp[channel,:]-np.mean(lfp[channel,:]), fs, nperseg=2000)
    PSD.append(Pxx)
PSD=np.vstack(PSD)  #matrix containing the power spectral density for each electrode (row), in the different frequencies (columns)

plt.figure()
plt.plot(f,PSD[0,:])
plt.xlabel('frequency [Hz]')
plt.title('PSD, channel 1, 30 day post implantation')
plt.xlim([0,60])
plt.ylabel('Power spectral density [\u03BCV^2]')    

MEAN_FREQ=[]
STD_FERQ=[]
freq_range=[[1,4],[4,8],[8,12],[12,30],[30,80],[80,200],[200,400],[400,750]]
for i in range(len(freq_range)):
        index_freq=np.where((f > freq_range[i][0]) & (f < freq_range[i][1]))[0]
        MEAN_FREQ.append(np.mean(PSD[:,index_freq]))
        STD_FERQ.append(np.std(PSD[:,index_freq]))
        
# MEAN_FREQ, STD_FERQ contain the mean and standard deviation of the PSD for each frequency band
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.title('Freq. range: ' + str(freq_range[i][0])+'-'+str(freq_range[i][1])+ ' Hz')
    plt.errorbar(30, MEAN_FREQ[i], yerr=STD_FERQ[i],
             elinewidth  = 2, ecolor='k', capsize = 4,
             fmt='o',  ms = 5, mec ='k', mew = '2' , mfc = 'w',
             zorder=2);
plt.subplot(3,3,8)
plt.xlabel('Days post surgery')
plt.subplot(3,3,4)
plt.ylabel('Power spectral density [\u03BCV^2]')    

#%% COHERENCE
#extract the coherence
coh=np.zeros([96,96,1001])
for i in range(96):
    for j in np.arange(i,96,1):
        f, Cxy = signal.coherence(lfp[i,:], lfp[j,:], fs,nperseg=2000)
        coh[i,j,:]=Cxy
        coh[j,i,:]=Cxy

#remove the symmetry in the distance matrix
for i in range(96):
    for j in np.arange(0,i,1):
        distance[i,j]=-1

#PLOT COHERENCE VS FREQUENCY   
plt.figure() 
plt.title('Day post implant: 30')    
distance_values=np.unique(distance)
coh_dist_mean=[]
coh_dist_std=[]
for dist in distance_values:
    if dist>=0:
        row,column=np.where(distance==dist)
        coh_dist=[]
        for i in range(len(row)):
            coh_dist.append(coh[row[i],column[i],:])
        index=np.where(np.sum(coh_dist,1)==0)[0]  #added to exclude bad channels
        coh_dist=np.delete(coh_dist,index,0)          #added to exclude bad channels
        coh_dist_mean.append(np.mean(coh_dist,0))
        coh_dist_std.append(np.std(coh_dist,0))    

cmap=plt.get_cmap('copper')
line_colors = cmap(np.linspace(0,1,50))
for i in range(len(coh_dist_mean)):
    if len(coh_dist_mean[i])>0:    #added to exclude bad channels
        plt.plot(f[:820],coh_dist_mean[i][:820],c=line_colors[i])            
plt.xlabel('Frequency [Hz]')
plt.ylabel('Coherence')
plt.ylim([0,1.01])
sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=distance_values[1],vmax=distance_values[-1]))
cbar=plt.colorbar(sm)
cbar.set_label('distance [mm]', rotation=90)

#PLOT COHERENCE VS DISTANCE
plt.figure()
plt.title('Day post implant: 30' )    
coh_dist_matrix = np.stack(coh_dist_mean)
freq_range=[[0,4],[4,8],[8,12],[12,30],[30,80],[80,200],[200,400]]
freq_range_str=['0-4','4-8','8-12','12-30','30-80','80-200','200-400']

cmap=plt.get_cmap('copper')
line_colors = cmap(np.linspace(0,1,7))
for i in range(len(freq_range)):
    index=np.where((f >= freq_range[i][0]) & (f < freq_range[i][1]))[0]
    app=coh_dist_matrix[:,index]
    app=np.mean(app,1)
    nanindex=np.where(np.isnan(app))[0] #added to exclude bad channels
    distanza=distance_values[1:]        #added to exclude bad channels
    distanza=np.delete(distanza, nanindex)  #added to exclude bad channels
    app=np.delete(app,nanindex) #added to exclude bad channels
    plt.plot(distanza,app,c=line_colors[i])
plt.xlabel('distance [mm]')
plt.ylabel('Coherence')
plt.ylim([0,1.01])
sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0,vmax=6))
cbar=plt.colorbar(sm)
cbar.set_ticks([0,1,2,3,4,5,6])
cbar.set_ticklabels([yi for yi in freq_range_str])
cbar.set_label('Frequency [Hz]', rotation=90)  

#COHERENCE MAP
coh_plot=np.mean(coh[:,:,16:24],2)  #between 8 and 12 Hz
coh_utah_map01(coh_plot,[])
plt.subplot(10,10,5)
plt.title('Coherence 8-12 Hz')
coh_plot=np.mean(coh[:,:,60:160],2)  #between 30 and 80 Hz
coh_utah_map01(coh_plot,[])
plt.subplot(10,10,5)
plt.title('Coherence 30-80 Hz')
#%% CORRELATION
lag=np.linspace(-2+1/fs,+2-1/fs,3999)
index_positive_lag=np.where(lag>=0)[0]

#calculate correlation
correlation=np.zeros([96,96,3999])
for i in range(96):
    for j in np.arange(i,96,1):
        lfp1=lfp[i,:]
        lfp2=lfp[j,:]
        
        #average over 2 seconds window. total of 28 window
        corr=[]
        for sec in np.arange(1,57,2):
            #we are centered in sec +1, we want 2 secnods of overlap
            l1=lfp1[sec*fs-fs:sec*fs+3*fs]
            l1 = (l1 - np.mean(l1)) / (np.std(l1) * len(l1))
            l2=lfp2[sec*fs-fs:sec*fs+3*fs]
            l2 = (l2 - np.mean(l2)) /  np.std(l2)
            co=np.correlate(l1,l2,'full')
            index=(len(co)-1)/4
            co=co[int(index):3*int(index)+1]
            corr.append(co)
        lag=np.linspace(-2+1/fs,+2-1/fs,3999)
        corr=np.mean(corr,0)
        correlation[i,j,:]=corr
        correlation[j,i,:]=np.flip(corr)    

corr=correlation

#PLOT CORRELATION VS LAGS at different distances   
plt.figure() 
plt.title('Day post implant: 30')    
distance_values=np.unique(distance)
corr_dist_mean=[]
corr_dist_std=[]
for dist in distance_values:
    if dist>=0:
        row,column=np.where(distance==dist)
        corr_dist=[]
        for i in range(len(row)):
                corr_dist.append(corr[row[i],column[i],:])
        index=np.where(np.sum(corr_dist,1)==0)[0]  #added to exclude bad channels
        corr_dist=np.delete(corr_dist,index,0)          #added to exclude bad channels
        corr_dist_mean.append(np.mean(corr_dist,0))
        corr_dist_std.append(np.std(corr_dist,0))


cmap=plt.get_cmap('copper')
line_colors = cmap(np.linspace(0,1,50))
for i in range(len(corr_dist_mean)):
    if len(corr_dist_mean[i])>0:
        plt.plot(lag[index_positive_lag],corr_dist_mean[i][index_positive_lag],c=line_colors[i])            
plt.xlabel('Lag [s]')
plt.ylabel('Correlation')
plt.ylim([-0.2,1.01])
sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=distance_values[1],vmax=distance_values[-1]))
cbar=plt.colorbar(sm)
cbar.set_label('distance [mm]', rotation=90)

#PLOT Correlation VS DISTANCE at different time lags
plt.figure()
plt.title('Day post implant: 30')    
corr_dist_matrix = np.stack(corr_dist_mean)
lag_range=[0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2]
index_lag=[1998, 2008,2048, 2098,2148,2198,2498, 2998, 3998]
lag_range_str=['0','0.01','0.05','0.1','0.15','0.2','0.5','1','2']

cmap=plt.get_cmap('copper')
line_colors = cmap(np.linspace(0,1,9))
for i in range(len(lag_range)):
    index=index_lag[i]
    app=corr_dist_matrix[:,index]
    nanindex=np.where(np.isnan(app))[0] #added to exclude bad channels
    distanza=distance_values[1:]        #added to exclude bad channels
    distanza=np.delete(distanza, nanindex)  #added to exclude bad channels
    app=np.delete(app,nanindex) #added to exclude bad channels
    plt.plot(distanza,app,c=line_colors[i])
plt.xlabel('distance [mm]')
plt.ylabel('Correlation')
plt.ylim([-0.2,1.01])
sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0,vmax=8))
cbar=plt.colorbar(sm)
cbar.set_ticks([0, 1,2,3,4,5,6,7,8])
cbar.set_ticklabels([yi for yi in lag_range_str])
cbar.set_label('Lag [s]', rotation=90)

#CORRELATION MAP
coh_plot=corr[:,:,1998]  #0 second lag
coh_utah_map01(coh_plot,[])
plt.subplot(10,10,5)
plt.title('0 lag')
coh_plot=corr[:,:,2048]  #0.1 second lag
coh_utah_map01(coh_plot,[])
plt.subplot(10,10,5)
plt.title('0.05 lag')

#%% PHASE COHERENCE
r_values=np.zeros([96,96,30])   #r is the phase coherence
rsurr_values=np.zeros([96,96,3000])

for i in range(96):
    for j in np.arange(i,96,1):
        lfp1=lfp[i,:]
        lfp2=lfp[j,:]
        
        lfp1H=signal.hilbert(lfp1)
        inst_phase1 =math.pi+np.angle(lfp1H)#inst phase
        lfp2H=signal.hilbert(lfp2)
        inst_phase2 = math.pi+np.angle(lfp2H)#inst phase
        
        R=[]
        R_surr=[]
        for sec in np.arange(0,60,2):
            ang1=inst_phase1[sec*fs:(2+sec)*fs]
            ang2=inst_phase2[sec*fs:(2+sec)*fs]
            diff=ang1-ang2
            index_neg=np.where(diff<0)[0]
            diff[index_neg]=2*math.pi+diff[index_neg]
            diff=diff-math.pi
            R.append(abs(np.mean(np.exp(1j*(diff)))))
            #generate surrogate data
            for iiii in range(100):
                shift=int(fs/2+np.random.rand()*fs/2)    #random time shift between 0.5 sec and 1 sec in samples
                ang_s=np.roll(ang2,shift)
                diff=ang1-ang_s
                index_neg=np.where(diff<0)[0]
                diff[index_neg]=2*math.pi+diff[index_neg]
                diff=diff-math.pi
                R_surr.append(abs(np.mean(np.exp(1j*(diff)))))
        
        r_values[i,j,:]=R
        r_values[j,i,:]=R
        rsurr_values[i,j,:]=R_surr
        rsurr_values[j,i,:]=R_surr

R=r_values
R_surr=rsurr_values

#PHASE COHERENCE MAP
coh_plot=np.mean(R,2)
coh_utah_map01(coh_plot,[])

#ALL R AND PLI
x_pos=[1, 2]
x_label=['R','R surr']
x_label2=['PLI','PLI surr']

a=np.triu(R[:,:,0],1)
index0=np.where(a==0)
R2=R
R_surr2=R_surr
for i in range(30):
    R2[:,:,i]=np.triu(R[:,:,i],1)
    R_surr2[:,:,i]=np.triu(R_surr[:,:,i],1)
R2=np.reshape(R2,[1,96*96*30]) 
R_surr2=np.reshape(R_surr2,[1,96*96*3000])    
index=np.where(R2!=0)
R2=R2[index]
index=np.where(R_surr2!=0)
R_surr2=R_surr2[index]
rmean=np.mean(R2)
rstd=np.std(R2)
rsurrmean=np.mean(R_surr2)
rsurrstd=np.std(R_surr2)

#R
fig, ax = plt.subplots()
ax.bar(x_pos, [rmean,rsurrmean], yerr=[rstd,rsurrstd], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_label)
ax.set_title('Day post implant: 30')


#R WITH RESPECT TO DISTANCE BETWEEN ELECTRODES
#remove the symmetry in the distance matrix
for i in range(96):
    for j in np.arange(0,i,1):
        distance[i,j]=-1
        
distance_values=np.unique(distance)
r_dist_mean=[]
r_dist_std=[]
rsurr_dist_mean=[]
rsurr_dist_std=[]
for dist in distance_values:
    if dist>=0:
        row,column=np.where(distance==dist)
        r_dist=[]
        rsurr_dist=[]
        for i in range(len(row)):
            r_dist.append(R[row[i],column[i],:])
            rsurr_dist.append(R_surr[row[i],column[i],:])

        index=np.where(np.sum(r_dist,1)==0)[0]  #added to exclude bad channels
        r_dist=np.delete(r_dist,index,0)          #added to exclude bad channels
        r_dist_mean.append(np.mean(r_dist))
        r_dist_std.append(np.std(r_dist))
        
        index=np.where(np.sum(rsurr_dist,1)==0)[0]  #added to exclude bad channels
        rsurr_dist=np.delete(rsurr_dist,index,0)          #added to exclude bad channels
        rsurr_dist_mean.append(np.mean(rsurr_dist))
        rsurr_dist_std.append(np.std(rsurr_dist))

plt.figure()
plt.plot(distance_values[2:],r_dist_mean[1:],'k')
plt.plot(distance_values[2:],rsurr_dist_mean[1:],'y')
plt.fill_between(distance_values[2:], np.asarray(r_dist_mean[1:])-np.asarray(r_dist_std[1:]), np.asarray(r_dist_mean[1:])+np.asarray(r_dist_std[1:]),alpha=0.2, edgecolor='k', facecolor='k') 
plt.fill_between(distance_values[2:], np.asarray(rsurr_dist_mean[1:])-np.asarray(rsurr_dist_std[1:]), np.asarray(rsurr_dist_mean[1:])+np.asarray(rsurr_dist_std[1:]),alpha=0.2, edgecolor='y', facecolor='y') 
plt.legend(['R','R surrogate'])    
plt.ylabel('Phase Coherence')
plt.xlabel('Distance [mm]')
plt.title('Day post implant: 30')

#%% 6 MONTHS TREND
#At the current stage, we can't provide the full code for the 6 months trend analysis, as it requires all the data for every single day.
# The statistic on the trend were obtained with the following command:

# slope,intercept,rvalue,pvalue,stderr=statistic.linregress(day_post_surgery,average_charachetristic_on_each_day)

#%% Pre post stimulation differences

# the code here show the difference between coherence before and after stimulation, the same code can be applied to the other measures after their extraction from the data

freq_range_str=['0-4','4-8','8-12','12-30','30-80','80-200','200-400']
freq_range=[[0,4],[4,8],[8,12],[12,30],[30,80],[80,200],[200,400]]

for i in range(96):
    for j in np.arange(0,i,1):
        distance[i,j]=-1
index04=np.where(distance==0.4)  

lfp = np.load('lfp_1KHz_30days_pre.npy')   #the data included are related to 30 days post implantation (before stimulation)
coh_pre=np.zeros([96,96,1001])
for i in range(96):
    for j in np.arange(i,96,1):
        f, Cxy = signal.coherence(lfp[i,:], lfp[j,:], fs,nperseg=2000)
        coh_pre[i,j,:]=Cxy
        coh_pre[j,i,:]=Cxy
        
lfp_post = np.load('lfp_1KHz_30days_post.npy')   #the data included are related to 30 days post implantation (after stimulation)  
coh_post=np.zeros([96,96,1001])
for i in range(96):
    for j in np.arange(i,96,1):
        f, Cxy = signal.coherence(lfp_post[i,:], lfp_post[j,:], fs,nperseg=2000)
        coh_post[i,j,:]=Cxy
        coh_post[j,i,:]=Cxy
        
pre_all=[]
post_all=[] 

plt.figure()
pi=[]
incremento=[]
for i in range(len(freq_range)):
    index=np.where((f >= freq_range[i][0]) & (f < freq_range[i][1]))[0]
    pre=np.mean(coh_pre[:,:,index],2)
    post=np.mean(coh_post[:,:,index],2)
    pre=pre[index04]
    post=post[index04]
    index0_pre=np.where(pre!=0)
    index0_post=np.where(post!=0)
    index0=np.intersect1d(index0_pre,index0_post)
    pre=pre[index0]
    post=post[index0]
    
    prem=np.mean(pre)
    pres=np.std(pre)
    postm=np.mean(post)
    posts=np.std(post)
    
    plt.errorbar(i*10-2, prem, yerr=pres,
                      elinewidth  = 2, ecolor=c1, capsize = 4,
                      fmt='o',  ms = 5, mec =c1, mew = '2' , mfc = 'w',
                      zorder=2);
    plt.errorbar(i*10+2, postm, yerr=posts,
                      elinewidth  = 2, ecolor=c2, capsize = 4,
                      fmt='o',  ms = 5, mec =c2, mew = '2' , mfc = 'w',
                      zorder=2);
    if len(pre>1):
        s1,p1=statistic.kstest(statistic.zscore(pre),'norm')
        s2,p2=statistic.kstest(statistic.zscore(post),'norm')

        if p1>0.05 and p2>0.05:
            s,p =statistic.ttest_rel(pre,post)
        else:
            s,p =statistic.wilcoxon(pre,post)
        
        if p<0.05:
            plt.plot(i*10,1.1*np.max([np.mean(pre)+np.std(pre),np.mean(post)+np.std(post)]),'*k')
    plt.xticks([0,10,20,30,40,50,60],freq_range_str)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.title('Day post implant: 30')
    pre_all.append(pre)
    post_all.append(post)
    pi.append(p)
    incremento.append(int((np.mean(post)-np.mean(pre))>0))
