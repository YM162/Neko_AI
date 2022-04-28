
import random
import os
import time
import neat
#import visualize
from statistics import mean
import pickle
from neat import statistics
import pandas as pd
import visualize
from sklearn import preprocessing
DINERO_INICIAL=100
import numpy as np
from random import randint


class Trader:
    def __init__(self,dinero_inicial):
        
        self.dinero=dinero_inicial
        self.portfolio=0
        self.valoranterior=dinero_inicial
        self.valor=dinero_inicial
        self.portfolioanterior=0
    def buy(self,precio):
        self.valoranterior=self.valor
        self.portfolioanterior=self.portfolio
        if self.dinero==0:
            return
        self.portfolio = self.dinero/precio
        self.dinero = 0
        self.valor=self.portfolio*precio
    def sell(self,precio):
        self.valoranterior=self.valor
        self.portfolioanterior=self.portfolio
        if self.portfolio==0:
            return
        self.dinero = self.portfolio*precio
        self.portfolio = 0
        self.valor=self.dinero

class Data:
    def __init__(self,path):
        self.data=pd.read_csv(path)
        self.precio=0
        self.dataclose=[]
        self.datavolume=[]
        self.datacloserel=[]
        self.datavolumerel=[]
        self.meanavg=[]
        self.meanavgposition=[]
        self.step=20
    def preprocess(self,steps):
        displacement=randint(0,97000)
        #print("Displacement: "+str(displacement))
        self.dataclose=list(self.data["close"])[displacement:displacement+steps]
        self.datavolume=list(self.data["volume"])[displacement:displacement+steps]
        
        for i in range(0,len(self.dataclose)):
            if i==0:
                self.datacloserel.append(0)
            else:
                self.datacloserel.append(((self.dataclose[i]/self.dataclose[i-1])-1))
        for i in range(0,len(self.datavolume)):
            if i==0:
                self.datavolumerel.append(0)
            else:
                self.datavolumerel.append(((self.datavolume[i]/self.datavolume[i-1])-1))
        self.datacloserel= np.array(self.datacloserel)
        self.datacloserel=self.datacloserel.reshape(-1, 1)
        self.datavolumerel= np.array(self.datavolumerel)
        self.datavolumerel=self.datavolumerel.reshape(-1, 1)
        
        self.datacloserel=preprocessing.normalize(self.datacloserel)
        self.datavolumerel=preprocessing.normalize(self.datavolumerel)
        self.datacloserel=list(self.datacloserel)
        self.datavolumerel=list(self.datavolumerel) 

        for i in range(0,len(self.dataclose)):
            if i<10:
                self.meanavg.append(self.dataclose[i])
            else:
                self.meanavg.append(mean(self.dataclose[i-10:i]))
        for i in range(0,len(self.dataclose)):
            if self.dataclose[i]>=self.meanavg[i]:
                self.meanavgposition.append(1)

            elif self.dataclose[i]<self.meanavg[i]:
                self.meanavgposition.append(-1)


    def next(self): #return False si sigue habiendo datos, return True si ya ha acabado
        #self.dataclose.pop(0)
        #self.datavolume.pop(0)
        #self.datacloserel.pop(0)
        #self.datavolumerel.pop(0)
        #self.meanavg.pop(0)
        #self.meanavgposition.pop(0)
        self.step+=1
        print("Step: "+str(self.step))
        if len(self.dataclose)==self.step:
            return True
        else:
            return False
    def get(self): #return vector con las 21 inputs 
        self.precio=self.dataclose[self.step]
        precio = self.precio
        vec1=self.datacloserel[self.step-10:self.step]
        vec2=self.datavolumerel[self.step-10:self.step]
        vec3=self.meanavgposition[self.step]
        vector=[]



        #vector.extend(vec1)
        #vector.extend(vec2)
        vector.append(vec3) 
        vector = tuple(vector+vec1+vec2)
        return precio,vector

def eval_genomes(genomes,config):
    global DINERO_INICIAL
    data=Data("datosETH.BTC5m.csv")
    data.preprocess(1000)

    nets = []
    traders = []
    ge = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        traders.append(Trader(DINERO_INICIAL))
        ge.append(genome)

    run = True
    while run and len(traders) > 0:
        
        print(len(traders))
        precio,inputs = data.get()
        print("Valor total: "+str(round(traders[0].valor,10))+"   Precio: "+str(precio))
        for x, trader in enumerate(traders): 
            output = nets[traders.index(trader)].activate(inputs) #Meter las inputs que quiera aquí.las primeras 20 correspondientes al precio,
                                                                  #las 20 siguientes al volumen, la distancia al rolling average, si está encima o 
                                                                  #debajo del rolling average
            
            if output[0] > 0:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                trader.buy(data.precio)

            if output[1] > 0:
                trader.sell(data.precio)

            for trader in traders:
                if trader.valor < DINERO_INICIAL*0.99:
                    #print(trader.valor)
                    ge[traders.index(trader)].fitness -= 1
                    nets.pop(traders.index(trader))
                    ge.pop(traders.index(trader))
                    traders.pop(traders.index(trader))

            for trader in traders:
                if trader.valor > trader.valoranterior:
                    ge[traders.index(trader)].fitness += 0.002

            for trader in traders:
                if trader.valor < trader.valoranterior:
                    ge[traders.index(trader)].fitness -= 0.002

            for trader in traders:
                if trader.portfolio == trader.portfolioanterior:
                    ge[traders.index(trader)].fitness -= 0.002

            
        
        if data.next() == True:
            for trader in traders:
                ge[traders.index(trader)].fitness += (trader.valor-99)**4
            #print("e")
            run = False
            break




def run(config_file):

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    with open('winner.pickle', 'wb') as f:
        pickle.dump(winner, f)
    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config,winner, view=True, filename="xor2-all.gv")



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)