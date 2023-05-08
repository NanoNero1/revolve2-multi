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
        self.configuration = [4,3,2]

        #This list is for accessing all the actors in a dataframe
        self.actFrame = pd.DataFrame(columns=['id', 'actor', 'preyPred','timeBorn'])
        self.actFrame.set_index('id')

        self.lastTime = (datetime.now().timestamp())

        cutIndex = math.ceil(len(self.actor_controllerList) / 2)
        
        #Initialize each actor_controller with a NN:
        for ind,actor in enumerate(self.actor_controllerList):
            actor.controllerInit(ind,
                                 self.new_denseWeights(self.configuration),
                                 ("prey" if ind <= cutIndex else "pred"),
                                 )
            list_row = [ind,actor,actor.preyPred,actor.timeBorn]
            self.actFrame.loc[len(self.actFrame)] = list_row

            self.actorCount += 1

        #print(self.actFrame)


        self.updPreyPred()

        header = ['id', 'predprey', 'tag', 'position']
        data = [
            ['Albania', 28748, 'AL', 'ALB'],
            ['Algeria', 2381741, 'DZ', 'DZA'],
            ['American Samoa', 199, 'AS', 'ASM'],
            ['Andorra', 468, 'AD', 'AND'],
            ['Angola', 1246700, 'AO', 'AGO']
                ]

        with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(data)

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
        return [ np.array(weights), np.array(biases)]
    
    #Allows us to make new mutated weight matrices from parents
    #alpha controls how harsh the mutations are
    def mutateWeights(self,weights,config,alpha=0.1):
        #Technically you could find the matrix size implicitly (and would be better design)
        mutWeights = weights.copy()
        for ind in range(len(config)-1):
            mutWeights[0][ind] += np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind],config[ind+1])) 
            mutWeights[1][ind] += np.random.uniform(low=-1.0*alpha, high=1.0*alpha, size=(config[ind+1],)) 
        return mutWeights
    
    #Combines two parents' genotpye to make a child genotype
    #STILL HAVE TO TEST!
    def crossover(self,parent1,parent2):
        parent1W = parent1.copy()
        parent2W = parent1.copy()
        crossWeights = []
        crossBiases = []

        for ind in range(len(self.configuration)-1):
            #Crossover Point at a random interval in the next layer
            cutIndex = randint(1,self.configuration[ind+1]-1) 

            crossWeights.append(np.concatenate(parent1W[0][ind][:cutIndex],parent2W[0][ind][cutIndex:]))
            crossBiases.append(np.concatenate(parent1W[1][ind][:cutIndex],parent2W[1][ind][cutIndex:]))
        return [crossWeights, crossBiases]

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

        #Passing info to the actor and asking it to control
        #Only pass info that is needed on every tick
        for ind, actor in enumerate(self.actor_controllerList):

            actor.passInfo(self.actorStates[ind],
                           self.get_grid_Tup(ind),
                           )
            actor.step(dt)
            actor_control.set_dof_targets(ind, actor.get_dof_targets())

        self.updateGrid()

        ## Time Based Section - doesnt update on every loop
        self.currTime = (datetime.now().timestamp())
        if float(self.currTime - self.lastTime) > 0.5:
            #print(self.currTime - self.lastTime)
            #print('predlist')
            #print(self.predList)
            #print(self.preyList)
            self.cognitiveActors(self.actorStates)
            self.writeMyCSV()
            print(f"prey: %s" % self.preyList.index)
            print(f"pred: %s" % self.predList.index)
            #print(self.actFrame.iloc[6])
            self.lastTime = (self.currTime)   

        #Loop for data collection
        if float(self.currTime - self.lastTime) > 0.3:
            raAct = randint(0,6)
            actor = self.actor_controllerList[raAct]

            datas = [actor.id,actor.preyPred,actor.tag,actor.bodyPos]
            #Normally, having a dynamically sized array especially with tons of data is a bad idea,
            #Luckily it gets emptied out fairly regularly so the ammortization isnt super harmful
            self.pushCollectData.append(datas)
            


    ###
    #Mechanics: this is where actors have their states changed according to the 
    #experiment setup ideas
    ###
        
    #Makes the brain switch from predator to prey and vice versa
    def switchBrain(self,id):
        print("change")
        actor = self.actor_controllerList[id]
        if actor.preyPred == "prey":
            #actor = self.actor_controllerList[self.preyList[id]]
            actor.preyPred = "pred"
            bestPred = self.bestGenotype("pred")

            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "pred"
        else:
            #actor = self.actor_controllerList[self.predList[id]]
            actor.preyPred = "prey"
            bestPred = self.bestGenotype("prey")

            #Update the actor dataframe
            self.actFrame.loc[id,"preyPred"] = "prey"

        #bestPred = self.new_denseWeights(self.configuration)
        
        actor.weights = self.mutateWeights(bestPred,self.configuration)
        itsNow = datetime.now().timestamp()
        actor.timeBorn = itsNow

        #Update the actor dataframe's time
        self.actFrame.loc[id,"timeBorn"] = itsNow

        

        self.updPreyPred()
        print(f"preyZO: %s" % self.preyList.index)
        print(f"predZO: %s" % self.predList.index)
        #Do something to setup new position??


    #Handles the mechanics of who gets caught and who dies out
    def updateGrid(self):
        self.updPreyPred()

        #Handles Death of Prey

        #All information retrieval needs to happen before changes are made
        #preyGrid = [(self.actor_controllerList[id]).gridID for id in self.preyList]
        preyGrid = [actor.gridID for actor in self.preyList["actor"]]
        #predTimes = ([actor.timeBorn for actor in self.actor_controllerList if actor.preyPred == "pred"])
        
        caught = None
        for pred in self.predList["actor"]:
            if caught != None:
                break
            #predGID = (self.actor_controllerList[pred]).gridID
            predGID = pred.gridID
            if predGID in preyGrid:
                caught = (self.preyList).index[preyGrid.index(predGID)]
            else:
                caught = None
            if caught != None and False:
                dumbo
                #print(caught)
                self.switchBrain(caught)
                #Hopefully this fixes it
                self.updPreyPred()
                preyGrid = [actor.gridID for actor in self.preyList["actor"]]
                #preyGrid = [(self.actor_controllerList[id]).gridID for id in self.preyList]

        #Handles Death of Predator
        minTime = min(self.predList["timeBorn"])
        #predID = predTimes.index(minTime)
        predID = self.predList["timeBorn"].idxmin()
            #print(pred.timeBorn)
            #print(self.lastTime)
            #print(pred.timeBorn - self.lastTime)

        #I don't know why but caught seems to activate despite no prey?
        if float(self.lastTime - minTime) > self.predatorlifeSpan() + 4.0 and True:
                #print(wenthere)
                self.switchBrain(predID)
    
    #Signals our robots to cognitively determine the next target angle
    def cognitiveActors(self,actorStates):
        #actorDistList = [actor.position]
        for ind,actor in enumerate(self.actor_controllerList):
            posList = [other.bodyPos for other in self.actor_controllerList]
            distList = [self.actorDist(actor.bodyPos,pos) for pos in posList]
            smallest = min(distList)
            closestActor = self.actor_controllerList[distList.index(smallest)]


            
            angle = self.angleBetween(actor.bodyPos,closestActor.bodyPos)
            dumbo = 2
            #This is where we can pass any cognitive information, 
            # right now it is: 0-angle 1-distance, 2-tag, 3-dumbo (test variable)
            #tag = randint(0,9)
            #print(tag)
            actor.makeCognitiveOutput(angle,smallest,closestActor.tag,dumbo)

            #(self.actorStates[0].position)[:2]

    #Get the LifeSpan of 
    def predatorlifeSpan(self):
        predsLeft = len(self.predList)
        
        if predsLeft > 1:
            #currently set to a linear scale
            return 4.0  + (20-predsLeft)
        else:
            return 1000000000

    ###
    #Informational Functions
    ###

    #Returns a tuple for where the actor is on the grid
    def get_grid_Tup(self, id):
        position = self.actorStates[id].position
        #NEED FIX: I dont super understand why its messing up with values other than 10
        x = round(position[0] * 0.5)
        y = round(position[1] * 0.5)
        return (x, y)
    
    #Get the oldest genotypes
    def bestGenotype(self,preyPred):
        #Update the predator and prey lists before checking, its probably uneccessary though
        self.updPreyPred()
        
        #actorMe
        #preyTimes = ([actor.timeBorn for actor in self.actor_controllerList if actor.preyPred == "prey"])
        #predTimes = ([actor.timeBorn for actor in self.actor_controllerList if actor.preyPred == "pred"])
        #preyTimes = self.preyList["timeBorn"]
        if preyPred == "prey":
            #maxTime = min(preyTimes)
            #preyID = preyTimes.index(maxTime)
            #genoID = self.preyList[preyID]
            genoID = self.preyList['timeBorn'].idxmin()
            
        else:
            #maxTime = max(predTimes)
            #predID = predTimes.index(maxTime)
            #genoID = self.predList[predID]
            genoID = self.predList['timeBorn'].idxmin()

        return (self.actor_controllerList[genoID]).weights

    #Updates which are prey and which are predators
    def updPreyPred(self):
        #self.preyList = [ actor.id for actor in self.actor_controllerList if actor.preyPred == "prey"]
        self.preyList = self.actFrame.query("preyPred=='prey'")
        self.predList = self.actFrame.query("preyPred=='pred'")
        #print(self.predList)
        #self.predList = [ actor.id for actor in self.actor_controllerList if actor.preyPred == "pred"]

    #Finds the distance between two actors, return a super large distance if same position
    #so that an actor "ignores" itself in terms of distance
    def actorDist(self,pos1,pos2):
        x = pos1[0] - pos2[0]
        y = pos1[1] - pos2[1]
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


    ###
    # Utility Functions
    ###

    def writeMyCSV(self):
        with open('countries.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write multiple rows
            writer.writerows(self.pushCollectData)

    #Updates the actor dataframe
    def updateActFrame(self):
        a=0

    

        




    




class Optimizer(EAOptimizer[Genotype, float]):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    _TERRAIN = terrains.flat()

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
        self._runner = LocalRunner(headless=False)

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

                print(f"fitness: {best_individual[1].value}")

                genotypem = (
                    await GenotypeSerializer.from_database(
                        session, [best_individual[0].genotype_id]
                    )           
                )[0]

            actor, controller = develop(genotype).make_actor_and_controller()
            #Number of actors found here
            #controllerList = [controller for i in range(4)]
            controllerList = []
            for i in range(8):
                actor, controller = develop(genotype).make_actor_and_controller()
                controllerList.append(controller)
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controllerList))
            env.static_geometries.extend(self._TERRAIN.static_geometry)

            #rng = np.random.default_rng()
            radius = 0.2
            engine = qmc.PoissonDisk(d=2, radius=radius)
            sample = engine.random(8)

            print(sample)

            for i in range(len(controllerList)):
                env.actors.append(
                    PosedActor(
                        actor,
                        Vector3(
                            [
                                sample[i][0]*3,
                                sample[i][1]*3,
                                bounding_box.size.z / 2.0 - bounding_box.offset.z + i*1,
                            ]
                        ),
                        Quaternion(),
                        [0.0 for _ in controller.get_dof_targets()],
                    )
                )    
            batch.environments.append(env)

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
        print(f"Fitness: %s " % float(end_state.position[1]*-1))
        return float(end_state.position[1]*-1)

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
