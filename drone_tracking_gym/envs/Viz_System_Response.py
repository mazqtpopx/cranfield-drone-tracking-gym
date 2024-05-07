from SecondOrderSystem import SecondOrderSystem
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

model = SecondOrderSystem(0.05, 1.0, 1.0, 0.0, 1.0)

drone_model = SecondOrderSystem(0.002, 0.2, 0.5, 0.0, 1.0)
#n_steps
n_steps_ptz = 40
n_steps_drone = 300

x_ptz=np.arange(0,n_steps_ptz,1) 
x_drone=np.arange(0,n_steps_drone,1) 

control = np.zeros((n_steps_ptz))
control[10:n_steps_ptz] = 1.0

control_drone = np.zeros((n_steps_drone))
control_drone[0:30] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[30:60] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[60:90] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[90:120] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[120:150] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[150:180] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[180:210] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[210:240] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[240:270] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50
control_drone[270:300] = np.random.uniform(low=-1.0, high=1.0, size=(1))[0] * 50

actual = np.zeros((n_steps_ptz))
actual_drone = np.zeros((n_steps_drone))
for n in range(n_steps_ptz):
    actual[n] = model.step(control[n])

for n in range(n_steps_drone):
    actual_drone[n] = drone_model.step(control_drone[n])


figsize =  (6,4)
plt.figure(figsize=figsize, dpi=300, layout='constrained')

plt.plot(x_ptz, control, color='royalblue', label='Target',)
plt.plot(x_ptz, actual, color='mediumseagreen', label='Response')

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.title(f"Response of the Camera Pan, Tilt, and Zoom")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
# plt.show()

plt.savefig(f"../../figures_journal/response_control_ptz.png")



figsize =  (6,4)
plt.figure(figsize=figsize, dpi=300, layout='constrained')

plt.plot(x_drone, control_drone, color='royalblue', label='Target',)
plt.plot(x_drone, actual_drone, color='mediumseagreen', label='Response')

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.title(f"Response of the Drone x,y,z Positions")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
# plt.show()

plt.savefig(f"../../figures_journal/response_control_drone.png")
