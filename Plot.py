# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:54:27 2019

@author: francisco
"""

import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
#import scipy.stats
#from DataFiles import DataFiles
#import classes.Variables as Variables

class Plot(object):

    
    def plotSerie(self, series, title, figNumber, resultsFolder):
        
        f = plt.figure(figNumber)   
        plt.rcParams.update({'font.size': 14})

        #dim = len(series[0,:])

        #color=plt.cm.rainbow(np.linspace(0,1,dim))
        #actions =  {'left'}
        #for i,c in zip(range(dim),color):
        plt.plot(series[:,0], label = 'Left', linestyle = '-', color = 'r')
        plt.plot(series[:,1], label = 'Right', linestyle = '-', color = 'b')
        plt.plot(series[:,2], label = 'Same', linestyle = '-', color = 'g')
            
#        plt.title(title)

        plt.legend(loc='best',prop={'size':10})

        plt.xlabel("Episodes")
#        plt.ylabel("Q-value")
#        plt.ylabel("Distance")
        plt.ylabel("Probability")
        plt.grid()
        plt.show()

        plt.xlim(0, 300)
        if figNumber != 2:
            plt.ylim(0, 1)
        plt.show()            
        #title.replace(" ", "")
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
        
    #end of plotSerie method
    
    def plot(self, serie1, serie2, title, resultsFolder, xlabel, ylabel, legendlabel1, legendlabel2, show = False):
        f = plt.figure()
        plt.rcParams.update({'font.size': 14})
        plt.plot(serie1, label = legendlabel1, linestyle = '--', color = 'r')
        # plt.plot(serie2, label = legendlabel2, linestyle = '--', color = 'b')
        
        convolveSet = 15
        convolveSerie1 = np.convolve(serie1, np.ones(convolveSet)/convolveSet)
        # convolveSerie2 = np.convolve(serie2, np.ones(convolveSet)/convolveSet)
        
        plt.plot(convolveSerie1, label = 'Smooth ' + legendlabel1, linestyle = '-', color = 'y')
        # plt.plot(convolveSerie2, label = 'Smooth ' + legendlabel2, linestyle = '-', color = 'g')
        plt.xlim(right=len(serie1))
        
        
        
        plt.title(title)

        plt.legend(loc='best',prop={'size':10})

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        if show:
            plt.show()  
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
    
    # def plotRewards(self, rewardsRL, rewardsIRL):
    #     [avgRewardsRL, rewardsRL, successRL] = self.averageData(rewardsRL, isFloat=True) 
    #     [avgRewardsIRL, rewardsIRL, successIRL] = self.averageData(rewardsIRL, isFloat=True)
    #     convolveSet = 30
    #     convolveAvgRewardsRL = np.convolve(avgRewardsRL, np.ones(convolveSet)/convolveSet)
    #     convolveAvgRewardsIRL = np.convolve(avgRewardsIRL, np.ones(convolveSet)/convolveSet)


    #     fig, ax = plt.subplots()

    #     tam = 16 #Fontsize
    #     plt.rcParams['font.size'] = tam
    #     plt.rc('xtick', labelsize=12) 
    #     plt.rc('ytick', labelsize=12) 
        
        

    #     ax.plot(avgRewardsIRL, label = 'Average reward IRL', linestyle = '--', color =  'r')
    #     ax.plot(avgRewardsRL, label = 'Average reward RL', linestyle = '--', color = 'y' )

    #     ax.plot(convolveAvgRewardsIRL, linestyle = '-', color =  '0.2')
    #     ax.plot(convolveAvgRewardsRL, linestyle = '-', color = '0.2' )
    #     #plt.plot(avgStepsRLAff, label = 'RL with affordances', marker = '.', linestyle = '-', color =  'b')

    #     ax.set_title('Collected reward')
    #     ax.legend(loc=4,prop={'size':tam-4})
    #     ax.set_xlabel('Episodes')
    #     ax.set_ylabel('Reward')
    #     ax.grid()

    #     # ax.set_ylim(Variables.punishment-0.8, Variables.reward)
    #     # ax.set_xlim(convolveSet, len(avgRewardsRL)/6)
    #     #my_axis = ax.gca()
    #     #ax.set_ylim(Variables.punishment-0.8, Variables.reward)
    #     #ax.set_xlim(convolveSet, len(avgRewardsRL)/6)
        
    #     plt.show()
        
    
    # def averageData(self, filename, isFloat=False):
    #     files = DataFiles()
    #     if isFloat:
    #         steps = files.readFloatFile(filename)
    #     else:
    #         steps = files.readFile(filename)
    #     #endif
    #     iterations = len(steps[0])
    #     tries = len(steps)
    
    #     avgSteps = np.zeros(iterations)
    #     success = np.zeros(iterations)
    
    #     for j in range(iterations):
    #         acum = 0.0
    #         cont = 0
    #         for i in range(tries):
    #             if steps[i][j] != 0:
    #                 acum += steps[i][j]
    #                 cont += 1
    #             #endif
    #         #endfor
    #         success[j] = cont
    #         if cont != 0:
    #             avgSteps[j] = acum / cont
    #         #endif
    #     #endfor
    #     return avgSteps, steps, success
    # #end of method averageData

    def plotQvalue(self, series, title, resultsFolder, show = True):
        
        f = plt.figure()   
        plt.rcParams.update({'font.size': 14})
        
        dim = len(series[0,:])
        # n = len(series[:,0])

        color=plt.cm.rainbow(np.linspace(0,1,dim))
        for i,c in zip(range(dim),color):
            plt.plot(series[:,i], label = 'Action '+str(i+1), linestyle = '-', color = c)
            # const_value = np.array([series[0,:] for i in range(n)])
            # plt.plot(const_value,linestyle='--')
            # print(c)

        #dim = len(series[0,:])

        #color=plt.cm.rainbow(np.linspace(0,1,dim))
        #actions =  {'left'}
        #for i,c in zip(range(dim),color):
        # plt.plot(series[:,0], label = '1', linestyle = '-', color = 'r')
        # plt.plot(series[:,1], label = '2', linestyle = '-', color = 'b')
        # plt.plot(series[:,2], label = '3', linestyle = '-', color = 'g')
        # plt.plot(series[:,3], label = '4', linestyle = '-', color = 'y')
        # plt.plot(series[:,4], label = '5', linestyle = '-', color = 'c')
        # plt.plot(series[:,5], label = '6', linestyle = '-', color = 'm')
        # plt.plot(series[:,6], label = '7', linestyle = '-', color = 'brown')
        # plt.plot(series[:,7], label = '8', linestyle = '-', color = 'yellowgreen')
        # plt.plot(series[:,8], label = '9', linestyle = '-', color = 'crimson')
        # plt.plot(series[:,9], label = '10', linestyle = '-', color = 'slateblue')
        # plt.plot(series[:,10], label = '11', linestyle = '-', color = 'gold')
        # plt.plot(series[:,11], label = '12', linestyle = '-', color = 'orange')
            
        plt.title(title)

        plt.legend(loc='best',prop={'size':10})

        plt.xlabel("Steps")
        plt.ylabel("Q-value")
#        plt.ylabel("Distance")
#        plt.ylabel("Probability")
        plt.grid()
        # plt.show(block = False)

#        plt.xlim(0, 300)
#        if figNumber != 2:
#            plt.ylim(0, 1)
        if show:
            plt.show()            
        #title.replace(" ", "")
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
        
    def plotAgentsProbability(self, series, title, resultsFolder, show = True):
        
        f = plt.figure()   
        plt.rcParams.update({'font.size': 14})
        
        # dim = len(series)
        # # n = len(series[:,0])

        # color=plt.cm.rainbow(np.linspace(0,1,dim))
        # for i,c in zip(range(dim),color):
        #     plt.plot(series[i], label = 'Agent '+str(i+1), linestyle = '-', color = c)
        convolveSet = 5
        convolveSerie = np.convolve(series, np.ones(convolveSet)/convolveSet)
        convolveSerie = convolveSerie[:len(series)]
        plt.plot(convolveSerie, label = 'Average', linestyle = '-', color = 'r')
        
        standard_dev = np.std(series)
        plt.fill_between(range(len(series)),convolveSerie + standard_dev, convolveSerie - standard_dev, color='r', alpha=0.2)
            
        plt.title(title)

        plt.legend(loc='best',prop={'size':10})

        plt.xlabel("Episodes")
        plt.ylabel("Probability")
#        plt.ylabel("Distance")
#        plt.ylabel("Probability")
        plt.ylim(0, 1.0)
        plt.xlim(0,len(series))
        plt.grid()
        # plt.show(block = False)

#        plt.xlim(0, 300)
#        if figNumber != 2:
#            plt.ylim(0, 1)
        if show:
            plt.show()            
        #title.replace(" ", "")
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
    


    def plotReward(self, serie, title, resultsFolder, show = True):
        f = plt.figure()   

        plt.plot(serie, label = 'rewards', linestyle = '-', color = 'r')
            
        plt.title(title)
        plt.legend(loc='best',prop={'size':10})
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        if show:
            plt.show()
        
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
    
    def plotRewardAgents(self, series, title, resultsFolder, smoothed = False, show = True):
        f = plt.figure()
        
        # plt.plot(serie, label = 'rewards', linestyle = '-', color = 'r')
        dim = len(series)
        # # n = len(series[:,0])

        color=plt.cm.rainbow(np.linspace(0,1,dim))
        
        if smoothed:
            convolveSet = 30
            convolveSerie = []
            for j in range(len(series)):
                # print("Serie")
                # print(serie)
                convolveSerie.append(np.convolve(series[j], np.ones(convolveSet)/convolveSet))
                # print("Convolve Serie")
                # print(serie)
            # print(convolveSerie)
            # print(series)
            for i,c in zip(range(dim),color):
                plt.plot(convolveSerie[i], label = 'Agent '+str(i+1), linestyle = '-', color = c)
            
            title = 'Smooth ' + title
        else:        
            for i,c in zip(range(dim),color):
                plt.plot(series[i], label = 'Agent '+str(i+1), linestyle = '-', color = c)
        
        plt.xlim(right=len(series[-1]))
        plt.title(title)
        plt.legend(loc='best',prop={'size':10})
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        if show:
            plt.show()
        
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
    
    def plotQvalueAgents(self, series, title, resultsFolder, smoothed = False, show = True):
        f = plt.figure()
        
        # plt.plot(serie, label = 'rewards', linestyle = '-', color = 'r')
        dim = len(series)
        # # n = len(series[:,0])

        color=plt.cm.rainbow(np.linspace(0,1,dim))
        
        if smoothed:
            convolveSet = 30
            convolveSerie = []
            for j in range(len(series)):
                # print("Serie")
                # print(serie)
                convolveSerie.append(np.convolve(series[j], np.ones(convolveSet)/convolveSet))
                # print("Convolve Serie")
                # print(serie)
            # print(convolveSerie)
            # print(series)
            for i,c in zip(range(dim),color):
                plt.plot(convolveSerie[i], label = 'Agent '+str(i+1), linestyle = '-', color = c)
            
            title = 'Smooth ' + title
        else:        
            for i,c in zip(range(dim),color):
                plt.plot(series[i], label = 'Agent '+str(i+1), linestyle = '-', color = c)
        
        plt.xlim(right=len(series[-1]))
        plt.title(title)
        plt.legend(loc='best',prop={'size':10})
        plt.xlabel("Episode")
        plt.ylabel("Q Value")
        plt.grid()
        if show:
            plt.show()
        
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
        
    def plotHardwareUsageAgents(self, x, y1, y2, legendy1, legendy2, labely1, labely2, title, resultsFolder,  error_y1 = None, error_y2 = None):
        fig, ax1 = plt.subplots()
        auxX= np.arange(len(x))
        width = 0.35
        ax2 = ax1.twinx()
        if error_y1 is not None:
            rects1 = ax1.bar(auxX - width/2, y1, yerr = error_y1, align = 'center', alpha = 0.6, ecolor='black', capsize = 10, width = width, label = legendy1, color= 'blue')
        else:
            rects1 = ax1.bar(auxX - width/2, y1, width = width, label = legendy1, color= 'blue')
        
        if error_y2 is not None:
            rects2 = ax2.bar(auxX + width/2, y2, yerr = error_y2, align = 'center', alpha = 0.6, ecolor='black', capsize = 10, width = width, label = legendy2, color = 'green')   
        else:
            rects2 = ax2.bar(auxX + width/2, y2, width = width, label = legendy2, color = 'green')
        
        ax1.set_xticks(auxX)
        ax1.set_xticklabels(x)
        ax1.set_ylabel(labely1)
        
        ax2.set_ylabel(labely2)
        
        plt.legend([rects1, rects2],[legendy1,legendy2])
        # ax2.legend(loc='best')
        ax1.set_title(title)
        
        ax1.set_ylim(0,max(y1)*1.3)
        ax2.set_ylim(0,100)
        
        autolabel(rects1,ax1)
        autolabel(rects2,ax2)
        plt.show()
        fig.savefig(resultsFolder + title + '.pdf', bbox_inches = 'tight')

    def plotSerieProbability(self, series, title, figNumber, resultsFolder):
        
        f = plt.figure(figNumber)   
        #dim = len(series[0,:])

        #color=plt.cm.rainbow(np.linspace(0,1,dim))
        #actions =  {'left'}
        #for i,c in zip(range(dim),color):
        plt.plot(series[:,0], label = 'in 8 actions', linestyle = '-', color = 'r')
        plt.plot(series[:,1], label = 'in 12 actions', linestyle = '-', color = 'b')
        plt.plot(series[:,2], label = 'in 16 actions', linestyle = '-', color = 'g')
            
        #plt.title(title)
        plt.legend(loc=4,prop={'size':10})
        plt.xlabel("Episodes")
        plt.ylabel("Probability")
        plt.grid()
        plt.show()

        plt.xlim(0, 100)
        plt.ylim(0, 1.05)
        plt.show()            
        #title.replace(" ", "")
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
        
    #end of plotSerie method
    

        #plot.plotSeries(numberOfTransitionsToGoalAcc, 'Number of Transitions to the Goal', 4, resultsFolder + str(experiment))

    def plotSeries(self, series, title, figNumber, resultsFolder):
        
        f = plt.figure(figNumber)   
        dim = len(series[0,:])

        color=plt.cm.rainbow(np.linspace(0,1,dim))
        for i,c in zip(range(dim),color):
            plt.plot(series[:,i], label = 'State '+str(i), linestyle = '-', color = c)
            
        #plt.title(title)
        #plt.legend(loc=4,prop={'size':10})
        plt.legend(loc='best',prop={'size':10})
        plt.xlabel("Episodes")
        plt.ylabel("Transitions")
        plt.grid()
        plt.show()

        plt.xlim(0, 150)
        plt.ylim(0, 30)
        plt.show()            
        #title.replace(" ", "")
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
        
    #end of plotSeries method


    def plotProbability(self, series, title, resultsFolder, show = True):
        
        f = plt.figure()
        dim = len(series[0,:])

        color=plt.cm.rainbow(np.linspace(0,1,dim))
        for i,c in zip(range(dim),color):
            plt.plot(series[:,i], label = 'Action '+str(i+1), linestyle = '-', color = c)


        # plt.plot(serie, label = '10 actions', linestyle = '-', color = 'r')
            
        plt.title(title)
        plt.legend(loc='best',prop={'size':10})
        plt.xlabel("Steps")
        plt.ylabel("Probability")
        plt.grid()
        if show:
            plt.show()

        # plt.xlim(0, 100)
        # plt.ylim(0, 1.1)
        # plt.show()            
        # title.replace(" ", "")
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
        
    #end of plotProbability method

    
    def plotHeatMap(self, matrix, title, figNumber, resultsFolder):
        f = plt.figure(figNumber)   
        plt.rcParams.update({'font.size': 18})
        plt.imshow(matrix, cmap='Oranges', interpolation='nearest', aspect='auto') #interpolation='bilinear'
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.xticks(np.arange(4), ('Down', 'Up', 'Right', 'Left'))
        plt.yticks(np.arange(11), np.arange(11))
        plt.show()
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
    #end of plotHeatMap method
    
    def correlation(self, data, title, figNumber, resultsFolder):
        f = plt.figure(figNumber) 
        plt.rcParams.update({'font.size': 14})
        corr = np.corrcoef(data, rowvar=False)
#        plt.matshow(corr)
        plt.imshow(corr, cmap='Oranges', interpolation='nearest', aspect='auto') #interpolation='bilinear'
        plt.colorbar()
#        plt.title(title)
        plt.xlabel("Actions")
        plt.ylabel("Actions")
        plt.xticks(np.arange(12), ('$L_m$', '$R_m$', '$S_m$', '$L_l$', '$R_l$', '$S_l$', '$L_p$', '$R_p$', '$S_p$', '$L_n$', '$R_n$', '$S_n$'))
        plt.yticks(np.arange(12), ('$L_m$', '$R_m$', '$S_m$', '$L_l$', '$R_l$', '$S_l$', '$L_p$', '$R_p$', '$S_p$', '$L_n$', '$R_n$', '$S_n$'))
        plt.show()

#        plt.title(title)
#        plt.colorbar()
        f.savefig(resultsFolder + title + '.pdf', bbox_inches='tight')
    #end of plotHeatMap method

#Q is the Q-value
#reward is the total reward collected 
#gamma is the discount factor
def pSuccess(Q, reward, gamma):
    n = np.log(Q/reward)/np.log(gamma) #corresponde a Eq. 6 del paper. Python no tiene logaritmo en base gamma, pero por propiedades de logaritmos se puede calcular en base 10 y dividir por el logaritmos de gamma. Es lo mismo, cualquier duda revisar propiedades de logaritmos. 
    log10baseGamma = np.log(10)/np.log(gamma) # Es un valor constante. Asumiendo que gamma no cambia. Se ocupa en la linea que viene a continuacion
    probOfSuccess = (n / (2*log10baseGamma)) + 1 #Corresponde a Eq. 7 del paper. Sin considerar la parte estocastica.
    # probOfSuccessLimit = np.minimum(1,np.maximum(0,probOfSuccess)) #Corresponde a Eq. 9 del paper. Lo mismo anterior, solo que limita la probabilidad a valores entre 0 y 1.
    #probOfSuccessLimit = probOfSuccessLimit * (1 - stochasticity) #Usar solo si usamos transiciones estocasticas o el parametro sigma
    return probOfSuccess # probOfSuccessLimit

#Normalize data 
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def autolabel(rects, ax):
    """Funcion para agregar una etiqueta con el valor en cada barra"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
if __name__ == "__main__":
    plot = Plot()

    resultsFolder = 'results/'
    
    # ------------------- First experiment--------------
    n_episodes = 500
    n_agents = 5
    
    # QvaluesAgents = []
    # ProbabilitiesLearningAvgAgents = []
    # ProbabilitiesIntrospectionAvgAgents = []
    # TotalRewardsAgents = []
    
    # for j in range(n_agents):
        
    #     agentFolder = 'agent' + str(j+1) + '/'
    
    #     QvaluesAvg = []
    #     # QvaluesAvg2 = []
    #     ProbabilitiesIntrospectionAvg = []
    #     ProbabilitiesLearningAvg = []
    
        
    #     QvaluesMax = []
    #     ProbabilitiesIntrospectionMax = []
    #     ProbabilitiesLearningMax = []
        
    #     for i in range(n_episodes):
    #         episodeFolder = 'episode' + str(i+1) + '/'
    #         Qvalues = np.loadtxt(resultsFolder + agentFolder + episodeFolder +'Qvalues.csv',delimiter=",")
    #         ProbabilitiesLearning = np.loadtxt(resultsFolder + agentFolder + episodeFolder +'ProbabilitiesLearning.csv',delimiter=",")
    #         ProbabilitiesIntrospection = np.loadtxt(resultsFolder + agentFolder + episodeFolder +'ProbabilitiesIntrospection.csv',delimiter=",")
            
            
    #         QvaluesAvg.insert(i,np.mean(Qvalues))
    #         qvaluemeans = Qvalues.mean(0)
    #         # print(qvaluemeans)
    #         # QvaluesAvg2.insert(i,qvaluemeans)
    #         ProbabilitiesLearningAvg.insert(i,np.mean(ProbabilitiesLearning))
    #         ProbabilitiesIntrospectionAvg.insert(i,np.mean(ProbabilitiesIntrospection))
            
    #         QvaluesMax.insert(i,np.max(Qvalues))
    #         ProbabilitiesLearningMax.insert(i,np.max(ProbabilitiesLearning))
    #         ProbabilitiesIntrospectionMax.insert(i,np.max(ProbabilitiesIntrospection))
        
        
    #     Rewards = np.loadtxt(resultsFolder + agentFolder + 'Reward.csv',delimiter=",")
        
    #     # plot.plot(pSuccess(np.array(QvaluesAvg),1500,0.95), pSuccess(np.array(QvaluesMax),1500,0.95), 'ProbabilityIntrospection-Reward1500', resultsFolder + agentFolder, 'Episodes', 'Probability', 'Average','Maximum')
        
    #     # Q Values
    #     plot.plot(QvaluesAvg, QvaluesMax,'Qvalues', resultsFolder + agentFolder,'Episodes','Qvalues', 'Average','Maximum')
        
    #     # Learning
    #     plot.plot(ProbabilitiesLearningAvg, ProbabilitiesLearningMax,'ProbabilitiesLearning',resultsFolder + agentFolder, 'Episodes','Probability', 'Average','Maximum')
        
    #     # Introspection
    #     plot.plot(ProbabilitiesIntrospectionAvg, ProbabilitiesIntrospectionMax,'ProbabilitiesIntrospection', resultsFolder + agentFolder, 'Episodes','Probability', 'Average','Maximum')
    #     #Normalize introspection probabilities
    #     ProbabilitiesIntrospectionAvg = NormalizeData(ProbabilitiesIntrospectionAvg)
    #     ProbabilitiesIntrospectionMax = NormalizeData(ProbabilitiesIntrospectionMax)
    #     plot.plot(ProbabilitiesIntrospectionAvg, ProbabilitiesIntrospectionMax,'ProbabilitiesIntrospection (Normalized)', resultsFolder + agentFolder, 'Episodes','Probability', 'Average','Maximum')
        
    #     # Rewards
    #     plot.plotReward(Rewards,'Rewards',resultsFolder + agentFolder)
        
    #     QvaluesAgents.insert(j, QvaluesAvg)
    #     ProbabilitiesLearningAvgAgents.insert(j,ProbabilitiesLearningAvg)
    #     ProbabilitiesIntrospectionAvgAgents.insert(j, ProbabilitiesIntrospectionAvg)
    #     TotalRewardsAgents.insert(j,Rewards)
    
    # plot.plotQvalueAgents(QvaluesAgents,'Q Values ' + str(n_agents) + '-agents',resultsFolder,smoothed=True)
    # ProbabilitiesLearningAvgAgents = np.array(ProbabilitiesLearningAvgAgents).mean(axis=0)
    # ProbabilitiesIntrospectionAvgAgents = np.array(ProbabilitiesIntrospectionAvgAgents).mean(axis=0)
    # plot.plotAgentsProbability(ProbabilitiesLearningAvgAgents,'Probabilities Learning ' + str(n_agents) + '-agents',resultsFolder)
    # plot.plotAgentsProbability(ProbabilitiesIntrospectionAvgAgents,'Probabilities Introspection ' + str(n_agents) + '-agents',resultsFolder)
    # plot.plotRewardAgents(TotalRewardsAgents,'Rewards ' + str(n_agents) + '-agents',resultsFolder,smoothed=True)
    
    # ------------------- Second experiment--------------
    memory_method, cpu_method = [], []
    error_memory, error_cpu = [], []
    
    # Learning method
    data_memory, data_cpu = [], []
    for i in range(6,9): # agent 6, 7 and 8
        agentFolder = 'agent' + str(i) + '/'
        
        memory = np.loadtxt(resultsFolder + agentFolder + 'Memory.csv',delimiter=",")
        memory = sum(memory/ 2**10)
        data_memory.append(memory)
        
        cpu = np.loadtxt(resultsFolder + agentFolder + 'Cpu.csv',delimiter=",")
        cpu = np.mean(cpu)
        data_cpu.append(cpu)
    
    print("l memory",data_memory)
    print("l cpu", data_cpu)
    memory_method.append(np.mean(data_memory))
    cpu_method.append(np.mean(data_cpu))
    
    error_memory.append(np.std(data_memory))
    error_cpu.append(np.std(data_cpu))
    
    # Introspection method
    data_memory, data_cpu = [], []
    for i in range(9,12): # agent 9, 10 and 11
        agentFolder = 'agent' + str(i) + '/'
        
        memory = np.loadtxt(resultsFolder + agentFolder + 'Memory.csv',delimiter=",")
        memory = sum(memory/ 2 ** 10)
        data_memory.append(memory)
        
        cpu = np.loadtxt(resultsFolder + agentFolder + 'Cpu.csv',delimiter=",")
        cpu = np.mean(cpu)
        data_cpu.append(cpu)
    
    print("i memory",data_memory)
    print("i cpu", data_cpu)
    memory_method.append(np.mean(data_memory) )
    cpu_method.append(np.mean(data_cpu))
    
    error_memory.append(np.std(data_memory))
    error_cpu.append(np.std(data_cpu))
    
    print("e memory", error_memory)
    print("e cpu", error_cpu)
    
    memory_method = np.round(memory_method, decimals=2)
    cpu_method = np.round(cpu_method, decimals=2)
    x = ["Learning", "Introspection"]
    plot.plotHardwareUsageAgents(x, memory_method, cpu_method, "Memory", "Cpu", "Memory (GB)", "Cpu (%)", "Hardware usage",resultsFolder, error_memory, error_cpu)
        
        
    
    
    
        # if i == 0:
        #     QvaluesAcc = Qvalues
        #     ProbabilitiesLearningAcc = ProbabilitiesLearning
        #     ProbabilitiesIntrospectionAcc = ProbabilitiesIntrospection
        #     RewardsAcc = Rewards
        # else:
        #     QvaluesAcc = QvaluesAcc + Qvalues
        #     ProbabilitiesLearningAcc = ProbabilitiesLearningAcc + ProbabilitiesLearning 
        #     ProbabilitiesIntrospectionAcc = ProbabilitiesIntrospectionAcc + ProbabilitiesIntrospection
        #     RewardsAcc = RewardsAcc + Rewards
    
    # QvaluesAcc = QvaluesAcc / n_episodes
    # ProbabilitiesLearningAcc = ProbabilitiesLearningAcc / n_episodes
    # ProbabilitiesIntrospectionAcc = ProbabilitiesIntrospectionAcc / n_episodes
    # RewardsAcc = RewardsAcc / n_episodes
    
    # plot.plotQvalue(QvaluesAcc,'Q values',1,resultsFolder)
    # plot.plotProbability(ProbabilitiesLearningAcc,'ProbabilitiesLearning',2,resultsFolder)
    # plot.plotProbability(ProbabilitiesIntrospectionAcc,'ProbabilitiesIntrospection',3,resultsFolder)
    # plot.plotReward(RewardsAcc,'Rewards',4,resultsFolder)
    
    # experiment="33"
    # state = 0
    
    # QEvoAcc = np.loadtxt(resultsFolder + "QValues" + experiment + ".csv", delimiter=',')
    # nFormAcc = np.loadtxt(resultsFolder + "nForm" + experiment + ".csv", delimiter=',')
    # PSuccessEvolutionAcc = np.loadtxt(resultsFolder + "PSuccessEvolution" + experiment + ".csv", delimiter=',')
    # PSuccessFormAcc = np.loadtxt(resultsFolder + "PSuccessForm" + experiment + ".csv", delimiter=',')
    # PChooseEvoAcc = np.loadtxt(resultsFolder + "PSuccessLearn" + experiment + ".csv", delimiter=',')

    # MXRLNoise = np.loadtxt(resultsFolder + "PSuccessNoisedMXRL" + experiment + ".csv", delimiter=',')
 
    # plot.plotSerie(QEvoAcc, 'Q-Values evolution from the initial state', 1, resultsFolder + str(experiment))
#    plot.plotSerie(nFormAcc, 'Estimated distance n to the reward', 2, resultsFolder + str(experiment))

    #Peter's correction of n value
#    nForm = np.log(QEvoAcc/Variables.REWARD)/np.log(Variables.GAMMA)
#    plot.plotSerie(nForm, 'Estimated distance n to the reward v2', 3, resultsFolder + str(experiment))
    
    # plot.plotSerie(PSuccessEvolutionAcc, 'Probability of success - Memory-based approach', 3, resultsFolder + str(experiment))
    # plot.plotSerie(PChooseEvoAcc, 'Probability of success - Learning-based approach', 4, resultsFolder + str(experiment))
    # plot.plotSerie(PSuccessFormAcc, 'Probability of success - Phenomenological-based approach', 5, resultsFolder + str(experiment))

#    plot.plotSerie(MXRLNoise, 'Probability of success - Noised MXRL approach', 6, resultsFolder + str(experiment))

    # matrixCorr = np.concatenate(([PSuccessEvolutionAcc, PChooseEvoAcc, PSuccessFormAcc, MXRLNoise]), axis=1)
#    plot.correlation(matrixCorr, 'Correlation', 7, resultsFolder + str(experiment))
    
    # mseLearning = (np.square(PSuccessEvolutionAcc - PChooseEvoAcc)).mean(axis=0)
    # msePhenomenological = (np.square(PSuccessEvolutionAcc - PSuccessFormAcc)).mean(axis=0)
    # mseMXRLNoise = (np.square(PSuccessEvolutionAcc - MXRLNoise)).mean(axis=0)
    
    # print(mseLearning)
    # print(msePhenomenological)
    # print(mseMXRLNoise)


    #Q-values
    #QValuesAcc= np.loadtxt(resultsFolder + "QValuesGridWorld30.csv", delimiter=',') 
    #QValuesAcc = np.delete(QValuesAcc, 11, 0)
    #QValuesScaled = 2 * (QValuesAcc/np.max(QValuesAcc) - 0.5)
    #QValuesScaled = QValuesAcc/np.max(QValuesAcc) 
    #plot.plotHeatMap(QValuesScaled, 'Q-Values', 0, resultsFolder + str(experiment))

    #PChoose
    #PChooseAcc = np.loadtxt(resultsFolder + "PChooseGridWorld30.csv", delimiter=',') 
    #PChooseAcc = np.delete(PChooseAcc, 11, 0)
    #plot.plotHeatMap(PChooseAcc, 'Probability of Choosing an Action', 1, resultsFolder + str(experiment))
    
    #PSuccess
    #PSuccessEvolutionAcc = np.loadtxt(resultsFolder + "PSuccessGridWorld31.csv", delimiter=',') 
    #PSuccessEvolutionAcc = np.delete(PSuccessEvolutionAcc, 11, 0)
    #plot.plotHeatMap(PSuccessEvolutionAcc, 'Probability of Success', 3, resultsFolder + str(experiment))

    #NTransiciones
#    numberOfTransitionsToGoalAcc = np.loadtxt(resultsFolder + "numberOfTransitionsToGoalGridWorld30.csv", delimiter=',') 
#    
#    numberOfTransitionsToGoalAcc = numberOfTransitionsToGoalAcc - 1
#
#    for ep in range(len(numberOfTransitionsToGoalAcc)):
#        for st in range(len(numberOfTransitionsToGoalAcc[0])):
#            if numberOfTransitionsToGoalAcc[ep][st] < 0:
#                numberOfTransitionsToGoalAcc[ep][st] = 0
#
#    
#    plot.plotSeries(numberOfTransitionsToGoalAcc, 'Number of Transitions to the Goal', 4, resultsFolder + str(experiment))

    #Probability of Success in Bounded grid world
#    numberOfActions = np.array([8, 12, 16])
#    #episode = 0
#    transitions = np.loadtxt(resultsFolder + "transitionsFrom031.csv", delimiter=',') 
#    #mean = np.mean(transitions[episode,:])
#    #std = np.std(transitions[episode,:])
#    #print(mean)
#    #print(std)
#    #normal = scipy.stats.norm(mean, std)
#    #print(normal.cdf(numberOfActions))
#    means = np.mean(transitions, axis=1)
#    stds = np.std(transitions, axis=1)
#    probabilityOfSuccess = np.zeros((len(means), len(numberOfActions)))
#    for i in range(len(means)):
#        for j in range(len(numberOfActions)):
#            normal = scipy.stats.norm(means[i], stds[i])
#            probabilityOfSuccess[i][j] = normal.cdf(numberOfActions[j])
#        
#    plot.plotSerieProbability(probabilityOfSuccess, 'Probability', 5, resultsFolder + str(experiment))
#






    #numberOfTransitionsToGoalAcc = np.loadtxt(resultsFolder+'numberOfTransitionsToGoalGridWorld30.csv', delimiter=',')
    #plot.plotSeries(numberOfTransitionsToGoalAcc, 'Number of Transitions to the Goal', 1, resultsFolder + str(30))

    #PChooseAcc = np.loadtxt("../PChooseGridWorldUnbounded.csv", delimiter=',')
    #numberOfTransitionsAcc = np.loadtxt("../numberOfTransitionsAccGridWorldUnbounded.csv", delimiter=',')

    #state = 0
    #PSuccessEvolutionAcc = np.loadtxt(resultsFolder + 'PSuccessRobotNavigationFromInitialStateEpisodes' + str(33) + '.csv', delimiter=",")
    #plot.plotSerie(PSuccessEvolutionAcc, 'Probability of Success from the Initial State', 5, resultsFolder + str(33))



    #print(QValuesAcc)
    #print(PChooseAcc)
    #print(numberOfTransitionsAcc)
    #plot.plotHeatMap(QValuesAcc, 'Q-Values', 10)
    #plot.plotHeatMap(PChooseAcc, 'Probability of Choose an Action', 11)
    #plot.plotHeatMap(numberOfTransitionsAcc, 'Number of Transitions', 13)


#end of class Plot

