"""Optimizer for finding a good modular robot body and brain using CPPNWIN genotypes and simulation using mujoco."""

import math
import pickle
from random import Random, randint
from typing import List, Tuple

import multineat
import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
import sqlalchemy
from genotype import Genotype, GenotypeSerializer, crossover, develop, mutate
from pyrr import Quaternion, Vector3
import quaternion as qt
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.optimization import DbId
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
#from revolve2.core.physics.environment_actor_controller import (
#    EnvironmentActorController,
#)
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import (
    ActorState,
    ActorControl,
    Batch,
    Environment,
    PosedActor,
    EnvironmentController,
    Runner,
)
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

#Dimitri Imports
#from tensorflow import keras
import numpy as np
from datetime import datetime
from scipy.stats import qmc
import csv
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
import pandas as pd
import warnings
with warnings.catch_warnings():
    warnings.warn("Let this be your last warning")
    warnings.simplefilter("ignore")
import random


# This is not exactly the same as the revolve class `revolve2.core.physics.environment_actor_controller.EnvironmentActorController`
class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controllerList: List[ActorController]

    def __init__(self, actor_controllerList: List[ActorController]) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for multiple actors in the environment.
        """
        self.actor_controllerList = actor_controllerList

        self.actorCount = 0
        self.cognitiveList = {}
        self.modelList = []
        self.configuration = [2,2,2]

        self.preyImm = 10

        #This list is for accessing all the actors in a dataframe
        self.actFrame = pd.DataFrame(columns=['id', 'actor', 'preyPred','timeBorn','lifeTime','gridID','lastKiller'])
        self.actFrame.set_index('id')

        self.lastTime = (datetime.now().timestamp())
        self.currTime = (datetime.now().timestamp())
        self.simStartTime = (datetime.now().timestamp())
        self.predDeathTime = (datetime.now().timestamp())
        self.preyDeathTime = (datetime.now().timestamp())
        self.lastTagTime = (datetime.now().timestamp())

        cutIndex = math.ceil(len(self.actor_controllerList) / 2)
        
        #Initialize each actor_controller with a NN:
        for ind,actor in enumerate(self.actor_controllerList):
            actor.controllerInit(ind,
                                 self.new_denseWeights(self.configuration),
                                 ("prey" if ind <= cutIndex else "pred"),
                                 )
            #gridID = self.get_grid_Tup(actor.id)
            list_row = [ind,actor,actor.preyPred,actor.timeBorn,actor.lifeTime,(0,0),actor.lastKiller]
            self.actFrame.loc[len(self.actFrame)] = list_row

            self.actorCount += 1

        self.updPreyPred()

        header = ['id', 'simTime', 'position','angle','predprey', 'tag',"otherID","immCheck","RanW",'closestAlly']
        data = [
            ['Albania', 28748, 'AL', 'ALB',1],
            ['Algeria', 2381741, 'DZ', 'DZA',1],
            ['American Samoa', 199, 'AS', 'ASM',1],
            ['Andorra', 468, 'AD', 'AND',1],
            ['Angola', 1246700, 'AO', 'AGO',1],
                ]

        with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            #writer.writerows(data)

        headerDeath = ['id', 'simTime','predprey', 'lifespan','RanW','caughtBy','byRanW']

        with open('deathBorn.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(headerDeath)

        self.pushCollectData = []

    ###
    # Neural Network Functions
    ###

    #Importing existing libraries was buggy, so I'm making my own neural net infrastructure
    #TO-DO: make smart NN initalization choices
    def new_denseWeights(self,config):
        weights = []
        biases = []
        for ind in range(len(config)-1):
            weights.append( np.random.uniform(low=-1.0, high=1.0, size=(config[ind],config[ind+1])) )
            biases.append( np.random.uniform(low=-1.0, high=1.0, size=(config[ind+1],)) )
        #jeez = np.array(weights,dtype=object)
        #return np.array([ weights, biases])
        return list([ weights, biases])
    
    #Allows us to make new mutated weight matrices from parents
    #alpha controls how harsh the mutations are
    def mutateWeights(self,weights,config,alpha=0.05):
        #Technically you could find the matrix size implicitly (and would be better design)
        mutWeights = weights.copy()
        for ind in range(len(config)-1):
            a = np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind],config[ind+1])) 
            mutWeights[0][ind] += a
            mutWeights[0][ind] = np.clip(mutWeights[0][ind],-1.0,1.0)
            b = np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind+1],)) 
            mutWeights[1][ind] += b
            mutWeights[1][ind] = np.clip(mutWeights[1][ind],-1.0,1.0)
        return mutWeights
    
    #Combines two parents' genotpye to make a child genotype
    #STILL HAVE TO TEST!
    def myCrossover(self,parent1,parent2):
        parent1W = np.copy(parent1)
        parent2W = np.copy(parent2)
        crossWeights = []
        crossBiases = []

        for ind in range(len(self.configuration)-1):
            #Crossover Point at a random interval in the next layer
            cutIndex = randint(1,self.configuration[ind+1]-1) 

            crossWeights.append(np.concatenate((parent1W[0][ind][:cutIndex],parent2W[0][ind][cutIndex:]))    )
            crossBiases.append(np.concatenate((parent1W[1][ind][:cutIndex],parent2W[1][ind][cutIndex:])))
        #return np.array([ crossWeights, crossBiases])
        return list([ crossWeights, crossBiases])

    ###
    #Control Section: This is the place where the cake is put together
    ###
    def control(self, dt: float, actor_control: ActorControl, argList: List) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        """

        self.actorStates = argList
        #print("baba")
        #print(len(argList))

        #Passing info to the actor and asking it to control
        #Only pass info that is needed on every tick
        #print("lol")
        #print([sta.position for sta in argList])
        #print("lol2")
        for ind, actor in enumerate(self.actor_controllerList):

            actor.passInfo(self.actorStates[ind],
                           self.get_grid_Tup(ind),
                           )
            actor.step(dt)
            actor_control.set_dof_targets(ind, actor.get_dof_targets())

        

        ## 
        

        ## Time Based Section - doesnt update on every loop

        
        self.currTime = (datetime.now().timestamp())

        if float(self.currTime - self.lastTime) > 2:
            

            for actor in self.actor_controllerList:
                if actor.immCheck == False and actor.preyPred == "prey":
                    posList = [other.bodyPos for other in self.predList["actor"]]
                    distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
                    smallest = min(distList)
                    if smallest > 5:
                        actor.immCheck = True
                    else:
                        actor.timeBorn = self.currTime


            self.updateGrid()
            
            self.cognitiveActors(self.actorStates)
            ##self.writeMyCSV()
            #print(f"prey: %s" % self.preyList.index)
            #print(f"pred: %s" % self.predList.index)
            #print(self.actFrame.iloc[0])
            self.positionMap()
            #print((self.actor_controllerList[0]).weights)
            self.lastTime = (self.currTime)   

        #Loop for data collection
        #if float(self.currTime - self.lastTime) > 0.3:
            #raAct = randint(0,0)
            ##actor = self.actor_controllerList[raAct]

            ##datas = [actor.id,actor.preyPred,actor.tag,actor.bodyPos]
            #Normally, having a dynamically sized array especially with tons of data is a bad idea,
            #Luckily it gets emptied out fairly regularly so the ammortization isnt super harmful
            ##self.pushCollectData.append(datas)
            


    ###
    #Mechanics: this is where actors have their states changed according to the 
    #experiment setup ideas
    ###
        
    #Makes the brain switch from predator to prey and vice versa
    def switchBrain(self,id):
        #print("change")
        actor = self.actor_controllerList[id]
        self.deathBornCSV(id,actor.preyPred,actor.timeBorn,actor.lastKiller)
        if actor.preyPred == "prey":
            #actor = self.actor_controllerList[self.preyList[id]]
            #lastKiller = actor.lastKiller if actor.lastKiller != None else 1
             


            if actor.lastPredWeights != None:
                #secondBest = self.actor_controllerList[actor.lastKiller].weights
                bestGeno = actor.lastPredWeights
            else:
                bestGeno = self.new_denseWeights(self.configuration)
            actor.preyPred = "pred"
            secondBest = self.bestGenotype("pred")

            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "pred"
            #print('joe')
            #joe = 0
            actor.immCheck = True
        else:
            #actor = self.actor_controllerList[self.predList[id]]
            #secondBest = self.new_denseWeights(self.configuration)
            #preyList = self.preyList["actor"]
            ##preyList = [actor for actor in self.preyList["actor"] if actor.immCheck ]
            ##posList = [other.bodyPos for other in preyList]
            ##distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
            ##smallest = min(distList)
            #closestPrey = self.actor_controllerList[(preyList.iloc[distList.index(smallest)]).id]
            ##closestPrey = preyList[distList.index(smallest)]
            #secondBest = closestPrey.weights
            ##genoID = list(random.choices(self.preyList.index, k=1, weights=(self.currTime - self.preyList['timeBorn']) ) )[0]
            ##secondBest = (self.actor_controllerList[genoID]).weights

            actor.preyPred = "prey"
            #bestGeno = self.bestGenotype("prey")
            if actor.closestPreyW != None:
                bestGeno = actor.closestPreyW
            else:
                bestGeno = self.new_denseWeights(self.configuration)
            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "prey"
            
            actor.immCheck = False
            #joe = 1
            #print('joe2')
            #print('doh')


        #Generation of new weights
        #actor.weights = self.mutateWeights(bestPred,self.configuration)
        #print(bestPred)
        #print('dd')
        #print(secondBest)
        reproChance = np.random.uniform(0.0,1.0)
        if reproChance < 0.2 and False:

            #Randomize Crossover Order to make sure not smae weights in every place
            if np.random.uniform(0.0,1.0) < 0.5:
                crossedOver = self.myCrossover(bestGeno,secondBest)
            else:
                crossedOver = self.myCrossover(secondBest,bestGeno)
            #print("ddd")
            #print(crossedOver)
            actor.weights = self.mutateWeights(crossedOver,self.configuration)
            actor.hasRanW = False
        elif reproChance < 0.66:
            actor.weights = self.mutateWeights(bestGeno,self.configuration)
            actor.hasRanW = False
        else:
            actor.weights = self.new_denseWeights(self.configuration)
            actor.hasRanW = True
        #print("dddd")
        #print(actor.weights)
        #if joe == 0:
        #    lol = 0
        #else:
        #    diddit
        #print("f")
        #print(actor.lifeTime)
        #print(actor.timeBorn)

        itsNow = datetime.now().timestamp()
        actor.timeBorn = itsNow
        actor.lifeTime = itsNow
        actor.lastTag = itsNow

        #Update the actor dataframe's time
        self.actFrame.loc[id,"timeBorn"] = itsNow
        self.actFrame.loc[id,"lifeTime"] = itsNow
        self.updateActFrame()
        self.updPreyPred()

    #Handles the mechanics of who gets caught and who dies out
    def updateGrid(self):
        #All information retrieval needs to happen before changes are made
        self.updateActFrame()
        self.updPreyPred()

        #THIS IS PROBABLY WHERE IT SLOWS DOWN
        #Handles Death of Prey


        
        caught = None
        pyL = self.preyList

        for prey in self.preyList["actor"]:
            preyList = [actor for actor in self.preyList["actor"] if actor.immCheck ]
            posList = [other.bodyPos for other in preyList]
            distList = [self.actorDist(prey.bodyPos,pos) for pos in posList]

            if len(distList) > 0:
                prey.smallAllyD = min(distList)
                prey.smallAllyID =(preyList[distList.index(prey.smallAllyD)]).id

        pdIndexes = [i for i in range(len(self.predList["actor"]))]
        #l_shuffled = random.sample(self.predList["actor"], len(self.predList["actor"]))
        l_shuffled = random.sample(pdIndexes, len(pdIndexes))
        #print(l_shuffled)
        #print('lol')
        #print([i.id for i in self.predList["actor"]])
        #print([i.id for i in self.preyList["actor"]])
        for predIND in l_shuffled:
            pred = self.predList["actor"].iloc[predIND]
            caught = None

            predList = [i for i in self.predList["actor"] if self.currTime - i.timeBorn > 30 ]
            posList = [other.bodyPos for other in predList]
            distList = [self.actorDist(pred.bodyPos,pos) for pos in posList]
            

            if len(distList) > 0:
                pred.smallAllyD = min(distList)
                pred.smallAllyID =(predList[distList.index(pred.smallAllyD)]).id
                #print(pred.id)
                #print(pred.smallAllyID)
                #dude

            #anti-stalling technique
            if len(self.preyList.index) < 7:
                break

            #if  len(self.predList.index) < 7:
            #    #posList = [other.bodyPos for other in predList]
            #    dWalls = 0
            #    for i,posit in enumerate(posList):
            #       myWalls = max(abs(posit[0]),abs(posit[1]))
            #        if myWalls > dWalls:
            #            dWalls = myWalls
            #            toKill = predList[i].id
            #    self.switchBrain([i.id for i in self.preyList["actor"]])
            #    self.switchBrain(toKill)
            #
            #    
            #    break
                

            #Separate theseeeee
            #caughtList = pyL[(pyL["gridID"] == pred.gridID) & (pred.id != pyL["lastKiller"]) & (pyL["timeBorn"] < self.currTime - 20)]
            #caughtList = pyL[(pred.id != pyL["lastKiller"]) & (pyL["timeBorn"] < self.currTime - self.preyImm)]
            
            #Distance based Implementation
            #preyList = caughtList["actor"]
            preyList = [actor for actor in self.preyList["actor"] if actor.immCheck ]
            posList = [other.bodyPos for other in preyList]
            distList = [self.actorDist(pred.bodyPos,pos) for pos in posList]
            
            if len(distList) > 0:
                smallest = min(distList)
            else:
                smallest = 1000

            if smallest < 1:
                #print(smallest)
                #print(pred.id)
                #caught = (preyList.iloc[distList.index(smallest)]).id
                caught = (preyList[distList.index(smallest)]).id
                #print(caught)


            #caught = caughtList.index[0] if len(caughtList) > 0 else None
            if caught != None:
                pred.closestPrey = None
                pred.closestPreyW = None
                #print(caught)
                #If a prey got close, but caught, it is a bad prey
                #if caught == pred.lastSeenPrey:
                #    pred.lastSeenPrey = None


                #A good predator can be born again
                ##pred.timeBorn = (datetime.now().timestamp())
                itsNow = (datetime.now().timestamp())
                pred.lifeTime = itsNow
                self.actFrame.loc[pred.id,"lifeTime"] = itsNow

                #print("ok")
                #print(self.actor_controllerList[caught].gridID)
                #print(pred.gridID)
                (self.actor_controllerList[caught]).lastPredWeights = pred.weights
                
                #Hopefully this fixes it
                self.actFrame.loc[caught,"lastKiller"] = pred.id
                self.actor_controllerList[caught].lastKiller = pred.id
                self.switchBrain(caught)
                
                
                
                self.updPreyPred()
                #
            else:
                lol = 0

        #Handles Death of Predator
        if len(self.predList) > 7 and (self.currTime - self.predatorlifeSpan() > self.predDeathTime):
            #print(self.predList["timeBorn"])

            #minTime = min(self.predList["lifeTime"])
            if np.random.uniform(0.0,1.0) < 1.0:
                predID = self.predList["lifeTime"].idxmin()
            else:
                smallDistG = 10
                for predIND in l_shuffled:
                    pred = self.predList["actor"].iloc[predIND]    

                    if pred.smallAllyD <= smallDistG:
                        smallDistG = pred.smallAllyD
                        predID = pred.smallAllyID
            #print(self.predList["lifeTime"])
            #print(predID)
            #print(self.predList.index)
            #print(predID)
            #haha
            self.switchBrain(predID)
            self.predDeathTime = (datetime.now().timestamp())
            #Main Conditional
            #if float(self.lastTime - minTime) > self.predatorlifeSpan() and True:
            #        #print(wenthere)
            #        self.switchBrain(predID)
        
        #A Random Prey May Die
        #if Flen(self.preyList) > 7 and (self.currTime - self.preylifeSpan() > self.preyDeathTime):
        if len(self.predList) <= 7:
            #minTime = min(self.preyList["lifeTime"])
            
            #preyID = self.preyList["lifeTime"].idxmin() 
            preyID = random.choice(self.preyList.index)
            self.switchBrain(preyID)
            #self.preyDeathTime = (datetime.now().timestamp())
    
    #Signals our robots to cognitively determine the next target angle
    def cognitiveActors(self,actorStates):
        for ind,actor in enumerate(self.actor_controllerList):
            #viableOther = list(filter(lambda other: ((actor.id != other.lastKiller) and ((other.timeBorn < self.currTime - self.preyImm) or (other.preyPred == 'pred'))), self.actor_controllerList))
            
            if actor.preyPred == "prey":
                viableOther = [pred for pred in self.predList["actor"] if (actor.tag == pred.tag) ]
            else:
                #viableOther = list(filter(lambda other: ((actor.id != other.lastKiller) and ((other.timeBorn < self.currTime - self.preyImm) and (other.preyPred == 'prey'))), self.actor_controllerList))
                viableOther = list(filter(lambda other: ((actor.tag == other.tag) and ((other.immCheck) and (other.preyPred == 'prey'))), self.actor_controllerList))

            posList = [other.bodyPos for other in viableOther]
            distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
            # I NEED TO FIX THIS make smallest huuuuge
            if len(distList) > 0:
                smallest = min(distList)
            else:
                continue

            closestActor = self.actor_controllerList[(viableOther[distList.index(smallest)]).id]
            if closestActor.preyPred == actor.preyPred:
                dudue

            actor.closestID = closestActor.id
            actor.smallDist = smallest
            #A prey that got close, but not caught is a good prey
            #if closestActor.preyPred == 'prey':
            #    actor.lastSeenPrey = closestActor.weights

            closestVector =  np.array(closestActor.bodyPos[:2]) - np.array(actor.bodyPos[:2])

            standardAngle = self.angleBetween(closestVector,[-1.0,-0.0])
            #print(actor.id)
            #print(actor.bodyA)
            #print(standardAngle)
            #goodAngle is still broken
            angle = self.goodAngle(actor.bodyA,standardAngle)
            #print(angle)
            #print(smallest)
            dumbo = 0
            #This is where we can pass any cognitive information, 
            # right now it is: 0-angle 1-distance, 2-tag, 3-dumbo (test variable)
            #tag = randint(0,9)
            #print(tag)

            #Normalizing inputs
            #angle = angle / math.pi
            #smallest = np.clip(smallest,-5,5) / 5
            angleMag = abs(angle) / math.pi
            
            
            if actor.preyPred == "prey":
                angle = self.modusAng(angle+math.pi)

            if angle > 0:
                LeftR = 1
            else:
                LeftR = -1

            angleNorm = angle / math.pi
            #print(actor.bodyA)
            #print(standardAngle)
            #print(angle)
            #print(angleMag)
            #print(LR)
            #print(actor.id)
            #print(closestActor.id)
            #dd
            isItPrey = 1 if closestActor.preyPred == "prey" else -1
            #print(isItPrey)

            #closer it gets, the more active it becomes
            inDist = np.clip(smallest/20,0.0,1.0)
            #inDist = (1/(1 + np.exp(-1*(0.5*smallest - 4))))

            if actor.preyPred == "pred":
                if actor.closestPrey != closestActor.id:
                    actor.closestPrey = closestActor.id
                    actor.closestPreyW = closestActor.weights

            dumbo = 0


            ## Get Ally Angle
            closestAlly = self.actor_controllerList[actor.smallAllyID]
            closestVector =  np.array(closestAlly.bodyPos[:2]) - np.array(actor.bodyPos[:2])
            standardAngle = self.angleBetween(closestVector,[-1.0,-0.0])
            angleAlly = (self.goodAngle(actor.bodyA,standardAngle)) / math.pi

            #You might be wondering why exactly are there 3 inputs, despite the methodology only saying 2
            #Due to some numpy array problems I can't seem to fix, ive added in a dummy variable always set to 0
            #So in the end it doesn't do anything, i.e., its still technically 2 inputs
            #print(actor.id)
            #print(actor.smallAllyID)
            #print(angleNorm)
            #print(angleAlly)
            actor.currTime = self.currTime
            #actor.makeCognitiveOutput(angleNorm,angleAlly,inDist,LR)
            actor.makeCognitiveOutput(LeftR,inDist)
            #actor.tarA = angle
            
            #actor.tarA = angle if isItPrey == 1 else self.modusAng(angle - math.pi)

            #actor.makeCognitiveOutput(angleNorm,closestActor.tag)
            #actor.makeCognitiveOutput(0,0)
            #(self.actorStates[0].position)[:2]

    #def getChannelList(tag,)

    #Get the LifeSpan of 
    def predatorlifeSpan(self):
        predsLeft = len(self.predList)
        
        if predsLeft > 2:
            #currently set to a linear scale
            return 50 - predsLeft*2
        else:
            return 1000000000
        
    def preylifeSpan(self):
        preysLeft = len(self.preyList)
        
        if preysLeft > 3:
            #currently set to a linear scale
            return 90 - 2*preysLeft
        else:
            return 1000000000

    ###
    #Informational Functions
    ###

    #Returns a tuple for where the actor is on the grid
    def get_grid_Tup(self, id):
        position = (self.actorStates[id].position)
        #NEED FIX: I dont super understand why its messing up with values other than 10
        x = round(position[0] * 0.2)
        y = round(position[1] * 0.2)
        return (x, y)
    
    #Get the oldest genotypes
    def bestGenotype(self,preyPred):
        #Update the predator and prey lists before checking, its probably uneccessary though
        self.updPreyPred()
        
        if preyPred == "prey":
            #genoID = self.preyList['timeBorn'].idxmin()
            bestDist = 0
            for prey in self.preyList['actor']:
                if prey.immCheck == False:
                    continue
                closestPos = (self.actor_controllerList[prey.closestID]).bodyPos
                closestDist = (prey.bodyPos[0] - closestPos[0])**2 + (prey.bodyPos[1] - closestPos[1])**2  
                if closestDist > bestDist:
                    bestDist = closestDist
                    genoID = prey.closestID      
        else:
            #genoID = self.predList['timeBorn'].idxmin()
            #A random lucky predator gets selected to reproduce
            #genoID = random.choice(self.predList.index) 
            genoID = list(random.choices(self.predList.index, k=1, weights=(self.currTime - self.predList['timeBorn']) ) )[0]

        return (self.actor_controllerList[genoID]).weights

    #Updates which are prey and which are predators
    def updPreyPred(self):
        self.preyList = self.actFrame.query("preyPred=='prey'")
        self.predList = self.actFrame.query("preyPred=='pred'")

    #Finds the distance between two actors, return a super large distance if same position
    #so that an actor "ignores" itself in terms of distance
    def actorDist(self,pos1,pos2):
        y = pos1[0] - pos2[0]
        x = pos1[1] - pos2[1]
        dist = math.sqrt( (x**2 + y**2) )
        if dist > 0.01:
            return dist
        else:
            return 10000000
        
    #This is the way to calculate angle WITH direction
    def angleBetween(self,v1,v2):
        dot = np.dot(v1,v2)                     #Dot Product
        det = (v1[0]*v2[1] - v2[0]*v1[1])       # Determinant
        angle = math.atan2(det, dot)            # atan2(y, x) or atan2(sin, cos)
        return angle

    def modusAngle(self,ang1,ang2):
        diff = (ang2+math.pi) - (ang1+math.pi)
        modused = (diff % (2*math.pi)) - math.pi
        return modused
    
    def goodAngle(self,ang1,ang2):
        modus = ang2 - ang1
        if abs(modus) > math.pi:
            modus += -2*math.pi*np.sign(modus)
        return modus

    def modusAng(self,ang):
        phase = ang + math.pi
        modused = (phase % (2*math.pi)) - math.pi
        return modused


    ###
    # Utility Functions
    ###

    def writeMyCSV(self):
        with open('countries.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write multiple rows
            writer.writerows(self.pushCollectData)

    #Here is all the output data needed for 
    def positionMap(self):

        with open('countries.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            simTime = self.currTime - self.simStartTime
            # write multiple rows
            for actor in self.actor_controllerList:
                newDataLine = [actor.id,simTime,actor.bodyPos,actor.bodyA,actor.preyPred,actor.tag,actor.closestID,actor.immCheck,actor.hasRanW,actor.smallAllyID] 
                writer.writerow(newDataLine)

    def deathBornCSV(self,id,preyPred,timeBorn,caughtBy):
        with open('deathBorn.csv', 'a', encoding='UTF8', newline='') as f:
            myActor = self.actor_controllerList[id]
            myBy = self.actor_controllerList[caughtBy]
            writer = csv.writer(f)
            lifespan = self.currTime - timeBorn
            simTimeNow = self.currTime - self.simStartTime
            # write multiple rows
            writer.writerow([id,simTimeNow,preyPred,lifespan,myActor.hasRanW,caughtBy,myBy.hasRanW])



    #Updates the actor dataframe
    def updateActFrame(self):
        #print([actor.bodyPos for actor in self.actor_controllerList])
        #print([actor.bodyA for actor in self.actor_controllerList])
        self.actFrame['gridID'] = [actor.gridID for actor in self.actor_controllerList]

    #def (self):

    

        
###     ###     ###     ###     ###     ###     ###     ###     ###
###     ###     ###     ###     ###     ###     ###     ###     ###
###################################################################
###################################################################
######################THE GREAT WALL OF CODE#######################
###################################################################
###################################################################
###################################################################

    




class Optimizer(EAOptimizer[Genotype, float]):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    #_TERRAIN = terrains.flat()
    #_TERRAIN = terrains.crater((20,20),1,1)
    _TERRAIN = terrains.jail()


    _db_id: DbId

    _runner: Runner

    _innov_db_body: multineat.InnovationDatabase
    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        initial_population: List[Genotype],
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        offspring_size: int,
    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param initial_population: List of genotypes forming generation 0.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database for the body genotypes.
        :param innov_db_brain: Innovation database for the brain genotypes.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        :param offspring_size: Number of offspring made by the population each generation.
        """
        await super().ainit_new(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
        )

        self._db_id = db_id
        self._init_runner()
        self._innov_db_body = innov_db_body
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
    ) -> bool:
        """
        Try to initialize this class async from a database.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database for the body genotypes.
        :param innov_db_brain: Innovation database for the brain genotypes.
        :returns: True if this complete object could be deserialized from the database.
        :raises IncompatibleError: In case the database is not compatible with this class.
        """
        if not await super().ainit_from_database(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
        ):
            return False

        self._db_id = db_id
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.db_id == self._db_id.fullname)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = opt_row.num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_body = innov_db_body
        self._innov_db_body.Deserialize(opt_row.innov_db_body)
        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(headless=True)

    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[float],
        num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                2,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[float],
        new_individuals: List[Genotype],
        new_fitnesses: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda n, genotypes, fitnesses: selection.multiple_unique(
                n,
                genotypes,
                fitnesses,
                lambda genotypes, fitnesses: selection.tournament(
                    self._rng, fitnesses, k=2
                ),
            ),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._innov_db_body, self._innov_db_brain, self._rng)

    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId,
    ) -> List[float]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )

        for genotype in genotypes:


            db = open_async_database_sqlite("./walkDatabase")
            async with AsyncSession(db) as session:
                best_individual = (
                    await session.execute(
                        select(DbEAOptimizerIndividual, DbFloat)
                        .filter(DbEAOptimizerIndividual.fitness_id == DbFloat.id)
                        .order_by(DbFloat.value.desc()))
                ).first()

                assert best_individual is not None

                #print(f"fitness: {best_individual[1].value}")

                genotype = (
                    await GenotypeSerializer.from_database(
                        session, [best_individual[0].genotype_id]
                    )           
                )[0]

            actor, controller = develop(genotype).make_actor_and_controller()
            #Number of actors found here
            #controllerList = [controller for i in range(4)]
            numberAGENTS = 30
            controllerList = []
            for i in range(numberAGENTS):
                actor, controller = develop(genotype).make_actor_and_controller()
                controllerList.append(controller)
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controllerList))
            env.static_geometries.extend(self._TERRAIN.static_geometry)

            #rng = np.random.default_rng()
            radius = 0.05
            engine = qmc.PoissonDisk(d=2, radius=radius)
            sample = engine.random(numberAGENTS)

            

            #print(sample)

            for i in range(len(controllerList)):
                env.actors.append(
                    PosedActor(
                        actor,
                        Vector3(
                            [
                                np.random.uniform(-1.0,1.0)*9*1,
                                np.random.uniform(-1.0,1.0)*9*1,
                                bounding_box.size.z / 2.0 - bounding_box.offset.z + i*0,
                            ]
                        ),
                        Quaternion(),
                        [0.0 for _ in controller.get_dof_targets()],
                    )
                )    
            batch.environments.append(env)
        #batch_results = await self._runner.
        batch_results = await self._runner.run_batch(batch)

        return [
            self._calculate_fitness(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0],
            )
            for environment_result in batch_results.environment_results
        ]

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        """ return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        ) """
        print(f"Fitness: %s " % float(end_state.position[0]*-1))
        return float(end_state.position[0]*-1)

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                db_id=self._db_id.fullname,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    """Optimizer state."""

    __tablename__ = "optimizer"

    db_id = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_body = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
