import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

import math

class IzhikevichNeuron:
    def __init__(self):
        self.a = 0.02
        self.b = 0.2
        self.c = -65 + 15 * np.random.rand(1)[0] ** 2
        self.d = 8 - 6 * np.random.rand(1)[0] ** 2

        self.v = -65
        self.u = self.b * self.v

        self.next_v = self.v
        self.next_u = self.u

        self.time = 0

        self.input_list = []
        self.weight_list = []

        self.fired = False
        self.fire_time = -1000
        self.presynaptic_fire_list = []

        self.alpha = 1
        self.thao = 20

    def register_input(self, input_neuron):
        self.input_list.append(input_neuron)
        self.weight_list.append(5 * np.random.rand(1)[0] + 10)

    def get_voltage(self):
        return self.v

    def get_fired(self):
        return self.fired

    def get_last_fire_time(self):
        return self.fire_time

    def get_synaptic_weight(self, index):
        return self.weight_list[index]

    def evaluate_state(self, external_input):

        total_input = external_input

        self.presynaptic_fire_list.clear()
        for i in range(len(self.input_list)):
            presynaptic_state = self.input_list[i].get_fired()
            if presynaptic_state:
                self.presynaptic_fire_list.append(i)
                total_input += self.weight_list[i]

        if self.fired:
            self.next_v = self.c
            self.next_u = self.u + self.d

        self.next_v += 0.04 * self.next_v ** 2 + 5 * self.next_v + 140 - self.next_u + total_input
        self.next_u += self.a * (self.b * self.next_v - self.next_u)

    def evaluate_synaptic_rule(self):
        for fired in self.presynaptic_fire_list:
            delta_time = self.fire_time - self.time - 1
            self.weight_list[fired] -= self.alpha * math.exp(delta_time / self.thao)

        if self.fired:
            for i in range(len(self.input_list)):
                delta_time = self.fire_time - self.input_list[i].get_last_fire_time()
                if delta_time > 0:
                    self.weight_list[i] += self.alpha * math.exp(-delta_time / self.thao)

    def update_state(self):
        self.v = self.next_v
        self.u = self.next_u

        if self.v > 30:
            self.fired = True
            self.fire_time = self.time
        else:
            self.fired = False

        self.time += 1


neuron1 = IzhikevichNeuron()
neuron2 = IzhikevichNeuron()

neuron2.register_input(neuron1)

membrane_voltage_1 = []
membrane_voltage_2 = []
synaptic_weight_12 = []
current_list1 = []
current_list2 = []

for i in range(1000):
    current_input1 = 5 * np.random.normal(0,1)
    current_input2 = 5 * np.random.normal(0, 1)

    current_list1.append(current_input1)
    current_list2.append(current_input2)
    membrane_voltage_1.append(neuron1.get_voltage())
    membrane_voltage_2.append(neuron2.get_voltage())
    synaptic_weight_12.append(neuron2.get_synaptic_weight(0))

    neuron1.evaluate_state(current_input1)
    neuron2.evaluate_state(current_input2)

    neuron1.evaluate_synaptic_rule()
    neuron2.evaluate_synaptic_rule()

    neuron1.update_state()
    neuron2.update_state()


time = range(1000)

# ax1 = plt.subplot(511)
# plt.plot(time, current_list1)
# plt.ylim(-30, 30)
#
# ax2 = plt.subplot(512, sharex=ax1)
# plt.plot(time, membrane_voltage_1)
# plt.ylim(-80, 50)
#
# ax3 = plt.subplot(513, sharex=ax1)
# plt.plot(time, current_list2)
# plt.ylim(-30, 30)
#
# ax4 = plt.subplot(514, sharex=ax1)
# plt.plot(time, membrane_voltage_2)
# plt.ylim(-80, 50)
#
# ax5 = plt.subplot(515, sharex=ax1)
# plt.plot(time, synaptic_weight_12)
#
# plt.show()

ax1 = plt.subplot(311)
plt.plot(time, membrane_voltage_1)
plt.ylim(-80, 30)

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(time, membrane_voltage_2)
plt.ylim(-80, 30)

ax3 = plt.subplot(313, sharex=ax1)
plt.plot(time, synaptic_weight_12)

plt.show()