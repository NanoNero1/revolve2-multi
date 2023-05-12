from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData
import math
#from revolve2.core.modular_robot import Body

#Dimitri Imports
import numpy as np
from datetime import datetime
#import time
from pyrr import Quaternion, Matrix33, matrix33, vector
import neat
#from tensorflow import keras
from random import randint

class CpgActorController(ActorController):
    """
    Cpg network actor controller.

    A state array that is integrated over time following the differential equation `X'=WX`.
    W is a weight matrix that is multiplied by the state array.
    The first `num_output_neurons` are the degree of freedom outputs of the controller.
    """

    _state: npt.NDArray[np.float_]
    _num_output_neurons: int
    _weight_matrix: npt.NDArray[np.float_]  # nxn matrix matching number of neurons
    _dof_ranges: npt.NDArray[np.float_]

    def __init__(
        self,
        state: npt.NDArray[np.float_],
        num_output_neurons: int,
        weight_matrix: npt.NDArray[np.float_],
        dof_ranges: npt.NDArray[np.float_],
        jointsLeft: List, 
        jointsRight: List,
    ) -> None:
        """
        Initialize this object.

        :param state: The initial state of the neural network.
        :param num_output_neurons: The number of output neurons. These will be the first n neurons of the state array.
        :param weight_matrix: The weight matrix used during integration.
        :param dof_ranges: Maximum range (half the complete range) of the output of degrees of freedom.
        """
        assert state.ndim == 1
        assert weight_matrix.ndim == 2
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert state.shape[0] == weight_matrix.shape[0]

        self._state = state
        self._num_output_neurons = num_output_neurons
        self._weight_matrix = weight_matrix
        self._dof_ranges = dof_ranges

        self._jointsLeft = jointsLeft
        self._jointsRight = jointsRight
        self.tarA = 0
        self.p = 4
        self.m33 = Matrix33()
        self.io = np.ndarray((2,), buffer=np.array([0.6,0.4]))
        self.getInfo = [0,0]

        self.timeBorn = datetime.now().timestamp()
        self.lastTime = self.timeBorn
        self.tag = 0
        self.lastKiller = None

    def step(self, dt: float) -> None:
        """
        Step the controller dt seconds forward.

        :param dt: The number of seconds to step forward.
        """
        self._state = self._rk45(self._state, self._weight_matrix, dt)

        #Updates our target angle
        #self.findTarAngle()

        #Time Loop that helps for debugging prints
        #         
        self.currTime = (datetime.now().timestamp())
        if float(self.currTime - self.lastTime) > 0.5:
            #print(f"Body Pos: %s" % self.bodyPos)
            #print(f"BAngle %s" % self.bodyA)
            #print(f"TAngle %s" % self.tarA)
            #print(f"L/R %s" % LR)
            #print(self.id)
            #print(self.gridID)

            #self.model_pred(np.ndarray((2,), buffer=np.array(self.getInfo)),self.weights)
            self.lastTime = self.currTime;



    ##Calculating angles
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    #I dont use this function anymore (and it's direction blind), but it might be useful in the future
    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)

        #signed_angle = math.atan2(v1_u[0]*v2_u[1]- v1_u[1]*v2_u[0],v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1])
        return (np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    #This is the correct way to to get the angle (pitch) from a quaternion
    def quat_to_angle(self, q):
        ang = math.atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
        return ang

    def findTarAngle(self):
        #This will be a neural network but for now its more simple 
        self.tarA = -1*self.bodyA - (math.pi/4.0)*1
        pass

    #Applies the ELU activation function for the NN
    def np_elu(self,x):
        mask = np.where(x<0)
        x[mask] = np.exp(x[mask])-1
        return x

    #Calculates the NN ouput manually with numpy
    def model_pred(self,input, weights):
        temp = input.copy()
        for weight, bias in zip(weights[0],weights[1]):
            #print(temp)
            #print("^t")
            #print(weight)
            #print("^w")
            #print(bias)
            #print("^b")
            temp = np.dot(temp, weight)+bias
            temp = self.np_elu(temp)
            #print(gotthru)

        #print(temp)
        return temp
    
    def makeCognitiveOutput(self,ang,dist,tag,dumbo):
        #There might be some reference issue here, check me
        #output = list(self.model_pred(np.ndarray((3,), buffer=np.array([ang,dist,tag])),self.weights)).copy()
        output = list(self.model_pred(np.array([ang,dist,tag,dumbo]),self.weights)).copy()
        self.tarA = output[0]

        #self.tag = round(np.clip(output[1],a_min=-1,a_max=1))
        self.tag = output[1]
        #print(f"this is tag %s " % self.tag)

    #Actor recieves real-time information here
    def passInfo(self, *args) -> None:
        actorState = args[0]
        ori = actorState.orientation
        self.m33 = Matrix33(matrix33.create_from_quaternion(ori))
        self.axis = actorState.orientation.axis

        self.bodyA = self.quat_to_angle(ori) - math.pi/4.0
        self.bodyPos = actorState.position
        
        self.gridID = args[1]
        #self.getInfo = args[2]
        pass

    #Initial instructions from the environment controller
    def controllerInit(self,id,weight_mat,preyPred):
        self.id = id
        self.weights = weight_mat
        self.preyPred = preyPred

    @staticmethod
    def _rk45(
        state: npt.NDArray[np.float_], A: npt.NDArray[np.float_], dt: float
    ) -> npt.NDArray[np.float_]:
        # TODO The scipy implementation of this function is very slow for some reason.
        # investigate the performance and accuracy differences
        A1: npt.NDArray[np.float_] = np.matmul(
            A, state
        )  # TODO matmul doesn't seem to be properly typed.
        A2: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A1))
        A3: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A2))
        A4: npt.NDArray[np.float_] = np.matmul(A, (state + dt * A3))
        return state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)

    def get_dof_targets(self) -> List[float]:
        """
        Get the degree of freedom targets from the controller.

        This will be the first `num_output_neurons` states from the state array.

        :returns: The dof targets.
        """

        #Copying means we are not referencing, we REALLY dont want to change self._state
        outPuts = self._state[0 : self._num_output_neurons].copy()

        scaleD = ((math.pi - abs(self.tarA))/math.pi)**self.p
        if self.tarA < 0:
            for i in self._jointsLeft:
                outPuts[i] = outPuts[i]*scaleD
                #outPuts[i] = 0
        else:
            for j in self._jointsRight:
                outPuts[j] = outPuts[j]*scaleD
                #outPuts[j] = 0
        
        return list(
            np.clip(
                outPuts,
                a_min=-self._dof_ranges,
                a_max=self._dof_ranges,
            )
        )

    def serialize(self) -> StaticData:
        """
        Serialize this object.

        :returns: The serialized object.
        """
        return {
            "state": self._state.tolist(),
            "num_output_neurons": self._num_output_neurons,
            "weight_matrix": self._weight_matrix.tolist(),
            "dof_ranges": self._dof_ranges.tolist(),
        }

    @classmethod
    def deserialize(cls, data: StaticData) -> CpgActorController:
        """
        Deserialize an instance of this class from `StaticData`.

        :param data: The data to deserialize from.
        :returns: The deserialized instance.
        :raises SerializeError: If this object cannot be deserialized from the given data.
        """
        if (
            not type(data) == dict
            or not "state" in data
            or not type(data["state"]) == list
            or not all(type(s) == float for s in data["state"])
            or not "num_output_neurons" in data
            or not type(data["num_output_neurons"]) is int
            or not "weight_matrix" in data
            or not type(data["weight_matrix"]) == list
            or not all(
                type(r) == list and all(type(c) == float for c in r)
                for r in data["weight_matrix"]
            )
            or not "dof_ranges" in data
            or not type(data["dof_ranges"]) == list
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        return CpgActorController(
            np.array(data["state"]),
            data["num_output_neurons"],
            np.array(data["weight_matrix"]),
            np.array(data["dof_ranges"]),
        )
