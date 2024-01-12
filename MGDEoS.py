# Import libraries
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import quad
import math
matplotlib.use('TkAgg')  # or another backend of your choice

# Import & read data
np.random.seed(9845)                                                      # For reproductivity; controls random number generation.
data = np.loadtxt("DymshitsPVT2014.txt")                                                     # Replace with your actual data file path.
data = data[data[:, 2].argsort()]

P_exp = data[:, 0]
P_error = data[:, 1]
T_exp = data[:, 2]
V_exp = data[:, 3]
V_error = data[:, 4]

A = 6.0221409e+23                                                                                             # Avogadro's Number.
Z = 8                                                              # Number of formula units. To be changed according to material.
kB = 1.380662e-23                                                                                       # Bolzmann's constant J/K.
No = 8                                                   # Number of atoms in the unit-cell. To be changed according to material.
R = 8.1345                                                                                                         # Gas constant.
thD0 = 550                                                               # Debye temperature. To be changed according to material.
T0 = 300                                                                                                  # Reference Temperature.

# SOLVE MODEL.
# 1. INITIAL MODEL. VALUES MUST CHANGE ACCORDING TO MATERIAL.
x = np.array([1450, 179, 6, 1, 1, 1])                                                                              # Initial model.
xl = np.array([1450, 10, 1, 0, -100, -100])                                                                          # Lower bound.
xu = np.array([1500, 500, 100, 100, 100, 100])                                                                       # Upper bound.

# 2. CREATE ANONYMOUS FUNCTION.
funMGDEoS = lambda x, V_exp=V_exp, T_exp=T_exp: (
    (
        (1.5 * x[1]) * (((x[0] / V_exp) ** (7 / 3)) - ((x[0] / V_exp) ** (5 / 3)))
        + (((9 / 8) * x[1]) * (x[2] - 4))
        * (
            (((x[0] / V_exp) ** (7 / 3)) - ((x[0] / V_exp) ** (5 / 3)))
            * (((x[0] / V_exp) ** (2 / 3)) - 1)
        )
    )
    + (x[3] * (T_exp - T0))
    + (x[4] * (-(T_exp - T0) * np.log(x[0] / V_exp)))
    + ((x[5] / 2) * ((T_exp - T0) ** 2))
) - P_exp

# 3. RUN LEAST-SQUARES
result = least_squares(funMGDEoS, x, bounds=(xl, xu), method='trf')

# 4. RETURN VALUES FOR THE UNKNOWN MODEL PARAMETERS.
print(f'V0: {result.x[0]}')
print(f'KT0: {result.x[1]}')
print(f'a0: {result.x[3] / result.x[1]}')
print(f"KT0_prime: {result.x[2]}")
print(f"aKT = (dP/dT)V: {result.x[3]}")
print(f"dKT/dT: {result.x[4]}")
print(f"d^2P/dT^2: {result.x[5]}")

v0 = result.x[0]
K0 = result.x[1]
K_prime = result.x[2]
aKT = result.x[3]
dKTdT = result.x[4]

# Mie-Gruneisen-Debye
MGDEoS = (
    (1.5 * result.x[1]) * (((result.x[0] / V_exp) ** (7 / 3)) - ((result.x[0] / V_exp) ** (5 / 3))) +
    (((9 / 8) * result.x[1]) * (result.x[2] - 4)) *
    (
        (((result.x[0] / V_exp) ** (7 / 3)) - ((result.x[0] / V_exp) ** (5 / 3))) *
        (((result.x[0] / V_exp) ** (2 / 3)) - 1)
    ) +
    (result.x[3] * (T_exp - T0)) +
    (result.x[4] * (-(T_exp - T0) * np.log(result.x[0] / V_exp))) +
    ((result.x[5] / 2) * ((T_exp - T0) ** 2))
)

# The thermal pressure, DPth
DPth_1 = (result.x[3] * (T_exp - T0)) + (
    result.x[4] * (-(T_exp - T0) * np.log(result.x[0] / V_exp))
) + ((result.x[5] / 2) * ((T_exp - T0) ** 2))

# convert (dKT/dT)V to (dKT/dT)P in order to extract KT:
dKTdT_P = result.x[4] - result.x[3] * result.x[2]
print(f"(dKT/dT)P: {dKTdT_P}")
KT = result.x[1] + (result.x[4] - result.x[3] * result.x[2]) * (T_exp - T0)

# We know from the thermal EoS that aKT = (dP/dT)V = gCV/V, where g the thermodynamic Gruneisen parameter.
# At high T, aKT ~constant, but a=a(P), i.e., it varies with pressure.
# The variation of a with P is directly related to the variation of KT with T.
# This is given by:
dadP_V = (1 / (KT ** 2)) * dKTdT_P

# For the two following equations, look at Wood et al (2008).
# These effects are more readily expressed via the Anderson-Gruneisen parameter, which takes both an isothermal and adiabatic forms.
# The isothermal delta_t0 = dKT/dP is defined by:
delta_t0 = (-1 / result.x[3]) * dKTdT_P
# At low pressures, delta_t is approximately constant, although structure-dependent.
# But at very high pressure, delta_t varies with pressure via:
delta_t = delta_t0 * (V_exp / result.x[0])

# PLOTTING.
# Step 1: CATEGORIZE EXPERIMENTAL DATA BASED ON TEMPERATURE
id = T_exp                                                  # Setting T as the id based on which the data file will be classified.
steps = P_exp                                                                                               # Steps include the P.
_, ib = np.unique(id, return_inverse=True)              # Identifying unique T values in the entire dataset to create the classes.
N = len(np.unique(id))                                                                                   # Number of id's present.
class_data = [data[ib == i, :] for i in range(1, N + 1)]
steps_count = [len(np.unique(steps[ib == i])) for i in range(1, N + 1)]

# Step 2: PLOT EXPERIMENTAL DATA.
plt.figure(1)
plt.gca().set_prop_cycle(None)                                                                                # Reset color cycle.
for i in range(len(class_data)):
    P = plt.errorbar(class_data[i][:, 0], class_data[i][:, 3], class_data[i][:, 4].astype(float), fmt='o', markersize=8)
plt.xlabel('P (GPa)', fontweight='bold')
plt.ylabel('V (Å$^3$)', fontweight='bold')
plt.show()

# Step 3: CATEGORIZE CALCULATED VALUES (i.e., MODEL).
Model = np.column_stack((MGDEoS, T_exp, V_exp, V_error))                                                  # Creating model matrix.
ID = Model[:, 1]                                       # Setting T as the id based on which the Model datafile will be classified.
step = data[:, 0]                                                                                           # Steps include the P.
_, ib = np.unique(ID, return_inverse=True)                  # Identifying unique T values in the entire dataset TO create classes.
N_ID = len(np.unique(ID))                                                                                # Number of id's present.
Class = [Model[ib == i, :] for i in range(1, N_ID + 1)]
step_count = [len(np.unique(step[ib == i])) for i in range(1, N_ID + 1)]

# Step 4: PLOT MODEL ON TOP OF EXPERIMENTAL. this is done by simply not creating a new figure.
for j in range(len(Class)):
    if len(Class[j]) > 0:
        Class[j] = Class[j][Class[j][:, 0].argsort()]                                        # Sort classes in case of randomness.
        plt.figure(1)
        plt.plot(Class[j][:, 0], Class[j][:, 2], 'k-', linewidth=1, label=f'Temperature {Class[j][0, 1]}K')
        plt.gca().set_prop_cycle(None)
plt.legend()
plt.show()
# -------------------------------------------------------------------------------------------------------------------------------

# Plotting the incompressibility, KT, vs pressure
plt.figure(2)
for h in range(len(class_data)):
    P = plt.plot(class_data[h][:, 0], (result.x[1] + (result.x[4] - result.x[3] * result.x[2]) * ((class_data[h][:, 2]) - T0)), 'o', markersize=8)
    plt.setp(P, markerfacecolor=plt.getp(P[0], 'color'))
    plt.xlabel('P (GPa)', fontweight='bold')
    plt.ylabel('$K_T$ (GPa)', fontweight='bold')
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')

# Plotting a/ao vs pressure
plt.figure(3)
for z in range(len(class_data)):
    P = plt.plot(class_data[z][:, 0], ((((class_data[z][:, 3] / result.x[0]) ** ((-1 / result.x[3]) * (dKTdT_P))) * (class_data[z][:, 3] / result.x[0]))), 'o', markersize=8)
    plt.setp(P, markerfacecolor=plt.getp(P[0], 'color'))
    plt.xlabel('P (GPa)', fontweight='bold')
    plt.ylabel(r'$\alpha/\alpha_0$ (K$^{-1}$)', fontweight='bold')
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')

# Plotting a vs temperature
plt.figure(4)
for zz in range(len(class_data)):
    P = plt.plot(class_data[zz][:, 2], ((result.x[3] / result.x[1]) * (((class_data[zz][:, 3] / result.x[0]) ** ((-1 / result.x[3]) * (dKTdT_P))) * (class_data[zz][:, 3] / result.x[0]))), 'o', markersize=8)
    plt.setp(P, markerfacecolor=plt.getp(P[0], 'color'))
    plt.xlabel('T (K)', fontweight='bold')
    plt.ylabel(r'$\alpha_0$ (K$^{-1}$)', fontweight='bold')
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')

# This part of the code contains a re-calculations of the MGDEoS to find the Gruneisen parameter γ assuming q=1. All unknown
# parameters are varied. But in reality V0, KT0, K'T0 should be equal to those derived above. This will implemented in the following
# lines.
# Define the function for integral
def integral_fun(z):
    return (z**3) / (np.exp(z) - 1)
# Solve integral of U(T)
T_int = thD0 / T_exp
f = np.zeros_like(T_int)
for i in range(len(T_int)):
    f[i], _ = quad(integral_fun, 0, T_int[i])
# Solve integral of U(T0)
T_int0 = thD0 / T0
f0, _ = quad(integral_fun, 0, T_int0)

# initial model
m = np.array([0.69, 1, 1485, 185, 4])
ml = np.array([0, 0, 200, 100, 0])
mu = np.array([2, 1, 1485, 187, 4])

# calculate internal energy
UT = 9 * kB * A * (N / Z) * T_exp * ((T_exp / thD0)**3) * f
UT0 = 9 * kB * A * (N / Z) * T0 * ((T0 / thD0)**3) * f0

# Define the function to minimize
def funPVTq(m):
    return (((1.5 * m[3]) * (((m[2] / V_exp)**(7 / 3)) - ((m[2] / V_exp)**(5 / 3))) *
             (1 + (3 / 4) * ((m[4] - 4) * (((m[2] / V_exp)**(2 / 3)) - 1)))) +
            (((m[0] * ((V_exp / m[2]) * m[1])) / V_exp) * (UT - UT0))) - P_exp

result = least_squares(funPVTq, m, bounds=(ml, mu))
# Fitted values
m = result.x
# Export fitted values
print('V0:', m[2])
print('KT0:', m[3])
print('g0:', m[0])
print('q:', m[1])
print('KT0_prime:', m[4])

PVTq = (((1.5 * m[3]) * (((m[2] / V_exp)**(7 / 3)) - ((m[2] / V_exp)**(5 / 3))) *
        (1 + (3 / 4) * ((m[4] - 4) * (((m[2] / V_exp)**(2 / 3)) - 1)))) +
       (((m[0] * ((V_exp / m[2]) * m[1])) / V_exp) * (UT - UT0)))
DPth_2 = (((m[0] * ((V_exp / m[2]) * m[1])) / V_exp) * (UT - UT0))
BM = ((3 / 2) * m[3]) * (((m[2] / V_exp)**(7 / 3)) - ((m[2] / V_exp)**(5 / 3))) * (1 + (3 / 4) * (m[4] - 4) * (((m[2] / V_exp)**(2 / 3)) - 1))

# plot the thermal pressure calculated from both methods to check the agreement.
# 1. for DPth_2 allowing all parameters vary.
plt.figure(5)
plt.plot(P_exp, DPth_1, label='DPth_1')
plt.plot(P_exp, DPth_2, label='DPth_2')
plt.xlabel('P_ (GPa)')
plt.ylabel('Thermal Pressure DPth (GPa)')
plt.legend()
plt.show()
# 2. for DPth_2 allowing only q and g0 parameters vary. But least-squares must run again.
# Define the function to minimize
j = np.array([0.69, 1])
jl = np.array([0, 0])
ju = np.array([2, 1])


def funPVTq1(j):
    return (((1.5 * K0) * (((v0/ V_exp)**(7 / 3)) - ((v0 / V_exp)**(5 / 3))) *
             (1 + (3 / 4) * ((K_prime - 4) * (((v0 / V_exp)**(2 / 3)) - 1)))) +
            (((j[0] * ((V_exp / v0) * j[1])) / V_exp) * (UT - UT0))) - P_exp

result = least_squares(funPVTq1, j, bounds=(jl, ju))
# Fitted values
j = result.x
print('g0:', j[0])
print('q:', j[1])

PVTq1 = (((1.5 * K0) * (((v0/ V_exp)**(7 / 3)) - ((v0 / V_exp)**(5 / 3))) *
             (1 + (3 / 4) * ((K_prime - 4) * (((v0 / V_exp)**(2 / 3)) - 1)))) +
            (((j[0] * ((V_exp / v0) * j[1])) / V_exp) * (UT - UT0)))
DPth_3 = (((j[0] * ((V_exp / v0) * j[1])) / V_exp) * (UT - UT0))

plt.figure(6)
plt.plot(P_exp, DPth_1, label='DPth_1')
plt.plot(P_exp, DPth_2, label='DPth_2')
plt.plot(P_exp, DPth_3, label='DPth_3')
plt.xlabel('P_ (GPa)')
plt.ylabel('Thermal Pressure DPth (GPa)')
plt.legend()
plt.show()

plt.figure(7)
plt.plot(P_exp, V_exp, 'o', markersize=8, label='data')
plt.plot(PVTq, V_exp, '*', markersize=5, label='Fit of PVTq to data')
plt.plot(PVTq1, V_exp, '.', markersize=5, label='Fit of PVTq1 to data')
plt.xlabel('P_ (GPa)')
plt.ylabel('Thermal Pressure DPth (GPa)')
plt.legend()
plt.show()

# OUTCOME: All three methods provide same values for the unknowns. PVTq and PVTq1 give almost identical q & g0.

# Assuming you have the required variables defined before this code block
# dKT_dP, gammao, delta_t0, m, V_exp, x, T_exp, KT, T0

# Calculating delta_s0
delta_s0 = delta_t0 - m[0]

# Gruneisen parameter
gamma = m[0] * ((V_exp / x[0])**m[1])
print(f' Gruneisen Parameter: {gamma}')

# Debye Temperature
result = np.exp((m[0] - gamma) / m[1])
thD1 = thD0 * result

# To calculate KS
a0 = aKT / K0
a = ((V_exp / v0)**delta_t0) * a0
KS = KT * (1 + a * gamma * T_exp)
KS0 = x[1] * (1 + a0 * m[0] * 1e-3 * T0)
print(f' KS0: {KS0}')

# Plotting the gruneisen parameter with temperature.
plt.figure(8)
plt.plot(T_exp, gamma, 'o')

