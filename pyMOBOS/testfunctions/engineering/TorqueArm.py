from ..TestFunction import TestFunction

class TorqueArm(TestFunction):

    def __call__(self, parameters):
        # Put parameters in form for FEA problem

        # Call FEA problem


        # Calculate Volume

        # Calculate max deflection


        # Ensure that torque arm didn't break (Check max stress for material)

        # Apply constraint penalty if torque arm did break

        # Return parameters to user


        pass

    def FEA_Torque_Arm(self, paramters):
        # Inputs: dimensions of the torque arm
        # Outputs: max deflection, volume

        # Call an FEA problem either through a Python package or 3rd party?

        # Return results....

        pass
