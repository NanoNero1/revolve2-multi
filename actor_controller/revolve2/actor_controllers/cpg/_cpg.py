from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData
import math
#from revolve2.core.modular_robot import Body
import numpy as np
from datetime import datetime
#import time
from pyrr import Quaternion, Matrix33, matrix33, vector

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
        self.p = 2
        self.m33 = Matrix33()
        #self.body = body

    def step(self, dt: float) -> None:
        """
        Step the controller dt seconds forward.

        :param dt: The number of seconds to step forward.
        """
        self._state = self._rk45(self._state, self._weight_matrix, dt)

        #This scales the joint activation functions to the target angles
        self.findTarAngle()
        scaleD = ((math.pi - abs(self.tarA))/math.pi)**self.p
        '''
        if self.tarA < 0:
            for i in self._jointsLeft:
                self._state[i] = self._state[i]*scaleD
        if self.tarA > 0:
            for j in self._jointsRight:
                self._state[j] = self._state[j]*scaleD
        '''
        if self.tarA < 0:
            for i in self._jointsLeft:
                self._state[i] = 0
        if self.tarA > 0:
            for j in self._jointsRight:
                self._state[j] = 0

        now = datetime.now()

        if now.microsecond % 20 < 1:
            # I need to figure out what does what
            # Vec1: "right"
            # Vec2: "up"
            # Vec3: z direction
            #so we use Vec1, consider "RIGHT as angle 0"

            #print(f"Body Pos: %s" % self.bodyPos)
            print(f"Body Vec1: %s" % self.m33.c1[:2])
            print(f"Body Vec2: %s" % self.m33.c2[:2])
            print(f"Body Vec3: %s" % self.m33.c3[:2])
            #print(f"Body A: %s" % self.bodyA)


    ##Calculating angles
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def findTarAngle(self):
        #This will be a neural network but for now its more simple 
        #self.tarA = self.angle_between(self.bodyA,[0,1])
        #self.tarA = -1*self.bodyA
        self.tarA = self.bodyA
        pass

    def passInfo(self, *args) -> None:
        actorState = args[0]
        ori = actorState.orientation
        self.m33 = Matrix33(matrix33.create_from_quaternion(ori))
        self.axis = actorState.orientation.axis
        self.bodyA = 1
        self.bodyPos = actorState.position
        pass


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
        return list(
            np.clip(
                self._state[0 : self._num_output_neurons],
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
