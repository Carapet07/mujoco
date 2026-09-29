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


def inverse_kinematics(x, y, phi, L1, L2):
    '''
    
        transforms desired trajectory from cartesian coordinates into joint angles which PID can read

        righ now it works with 3 joint arm

        phi - desired orientation
    '''

    # check if the target is in robot's arm length reach 
    distance = np.hypot(x, y)
    tolerance = 1e-9
    if (distance > L1+L2 + tolerance or
            distance < abs(L1 - L2) + tolerance):
        raise ValueError('Target position is out of reach ')
    
    # elbow angle
    c2 = (x*x + y*y - L1*L1 - L2*L2) / (2 * L1 * L2)

    # numerical safety
    c2 = np.clip(c2, -1.0, 1.0)
    # convert into elbow angles in radians
    q2 = np.arccos(c2)

    # shoulder angle
    q1 = np.arctan2(y, x) - np.arctan2(
        L2 * np.sin(q2),
        L1 + L2 * np.cos(q2)
    )

    # wrist angle
    q3 = phi - q1 - q2

    return np.array([q1, q2, q3])



# --------------------------- function not used, maybe will be deleted -------------------------------------
#  define our trajectory (sinusoidal wave)
# t - time in seconds since the simulation loop started
def desired_trajectory(t: float):
    """ 
        !! DO NOT USE WHEN inverse_kinematics IS BEING CALLED
        produces the desired joint positions
        sine and cos move between -1 and 1, amplitude of 0.5 shrinks that range 
        down to -0.5 to 0.5

    """ 
    amplitude = 6.24


    joint1_desired = amplitude * np.sin(1*np.pi*t) # sinusoidal trajectorty, w = pi
    joint2_desired = amplitude * np.cos(1*np.pi*t)
    joint3_desired = 0 * np.sin(1*np.pi*t)
    return joint1_desired, joint2_desired, joint3_desired




# controls joint angles
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
def mujoco_launcher(target_pos):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        prev_error = [0, 0, 0]
        integral = [0, 0, 0]


        trajectory1 = []
        trajectory2 = []
        trajectory3 = []
        signals1 = []
        signals2 = []
        signals3 = []
        while viewer.is_running():
            current_time = time.time() - start_time
            dt = model.opt.timestep


            # get joint possitions
            cur_pos = data.qpos[:3].copy() # 3 jonits


            # Compute control signals
            control_signals = []
            for i in range(3):
                command, prev_error[i], integral[i] = pid_control(
                    target_pos[i],
                    cur_pos[i],
                    prev_error[i],
                    integral[i],
                    dt,
                )
                control_signals.append(command)

            data.ctrl[:] = control_signals
            
            # log data
            trajectory1.append(target_pos[0])
            trajectory2.append(target_pos[1])
            trajectory3.append(target_pos[2])
            signals1.append(control_signals[0])
            signals2.append(control_signals[1])
            signals3.append(control_signals[2])
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
            data.ctrl[:] = control_signals  

            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            time.sleep(dt)


if __name__ == '__main__':
    # specify cartesian coordinates
    target_pos = inverse_kinematics(
        x=0.2,
        y=0.3,
        phi=0,
        L1 = 0.3,
        L2 = 0.3
    )
    
    run_mujoco = mujoco_launcher(target_pos)

