import mujoco 
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time


# load xml model

# create model and data objects of xml robot
model = mujoco.MjModel.from_xml_path("robot.xml")  # robot's structure, physics settings
data = mujoco.MjData(model) # contains robot's changing state: joint pos, vels etc



Kp = 25.0 # proportional gain
Ki = 0.1 # integral gain
Kd = 1.0 # derviative gain


#  define our trajectory (sinusoidal wave)
# t - time in seconds since the simulation loop started
def desired_trajectory(t: float):
    """ 
        produces the desired join positions
        sine and cos move between -1 and 1, amplitude of 0.5 shrinks that range 
        down to -0.5 to 0.5

    """ 
    amplitude = 0.5
    joint1_desired = 0.5*amplitude * np.sin(1*np.pi*t) # sinusoidal trajectorty, w = pi
    joint2_desired = 0.5*amplitude * np.cos(1*np.pi*t)
    return joint1_desired, joint2_desired


def pid_control(target, current, prev_error, integral, dt):
    '''
        target angle
        current angle
        dt - time between two steps
    '''

    error = target - current
    integral += error * dt  # accumulated error over time
    derivative = (error - prev_error) / dt    
    control_signal = Kp * error + Ki * integral + Kd * derivative
    return control_signal, error, integral


# Launch the simulation 
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    prev_error = [0, 0]
    integral = [0, 0]


    trajectory1 = []
    trajectory2 = []
    signals1 = []
    signals2 = []
    while viewer.is_running():
        current_time = time.time() - start_time
        dt = model.opt.timestep


        # get joint possitions
        cur_pos = [data.qpos[0], data.qpos[1]] # 2 jonits
        target_pos = desired_trajectory(current_time)


        # Compute control signals
        control_signals = []
        for i in range(2):
            control_signal, prev_error[i], integral[i] = pid_control(target_pos[i], cur_pos[i], prev_error[i], integral[i], dt)
            control_signals.append(control_signal)

        
        # log data
        trajectory1.append(target_pos[0])
        trajectory2.append(target_pos[1])
        signals1.append(control_signals[0])
        signals2.append(control_signals[1])
        x = np.linspace(0, 10, 500)

        # CHANGE TO True TO TRIGGER LOGGING
        if ((False) and (len(trajectory1) == 500)):
            plt.plot(x, trajectory1)
            plt.plot(x, trajectory2)
            plt.plot(x, signals1)
            plt.plot(x, signals2)
            plt.title("Signals and trajectories until over 1000 timesteps")
            plt.show()

        # Apply control signals
        data.ctrl[0] = control_signals[0]  # First actuator controls first joint
        data.ctrl[1] = control_signals[1]  # Second actuator controls second joint
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Update viewer
        viewer.sync()
        time.sleep(dt)
