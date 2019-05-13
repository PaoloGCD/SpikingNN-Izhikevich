import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

import math
from time import sleep

class IzhikevichExitatoryNeuron:
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

time = 1000
image_height = 32
image_width = 32


# Define network structure
# One exhitatory layer 32x32 neurons, each neuron is interconnected to its 3x3 neighborhood
sensory_layer = []
for j in range(image_height+2):
    row = []
    for i in range(image_width+2):
        row.append(IzhikevichExitatoryNeuron())
    sensory_layer.append(row)

for j in range(1,image_height+1):
    for i in range(1,image_width+1):
        sensory_layer[j][i].register_input(sensory_layer[j-1][i-1])
        sensory_layer[j][i].register_input(sensory_layer[j-1][i])
        sensory_layer[j][i].register_input(sensory_layer[j-1][i+1])
        sensory_layer[j][i].register_input(sensory_layer[j][i-1])
        sensory_layer[j][i].register_input(sensory_layer[j][i+1])
        sensory_layer[j][i].register_input(sensory_layer[j+1][i-1])
        sensory_layer[j][i].register_input(sensory_layer[j+1][i])
        sensory_layer[j][i].register_input(sensory_layer[j+1][i+1])

# Prepare plotting window
pt = plt.imshow(-65*np.ones(image_height*image_width).reshape(image_height,image_width), vmin=-80, vmax=30)
plt.colorbar(orientation='vertical')

# Run network
layer_voltage_list = []
for t in range(time):

    # Process input
    neuron_voltage_list = []
    for j in range(1, image_height+1):
        for i in range(1, image_width+1):
            sensory_layer[j][i].evaluate_state(5 * np.random.normal(0,1))
            sensory_layer[j][i].evaluate_synaptic_rule()

            neuron_voltage_list.append(sensory_layer[j][i].get_voltage())

    layer_voltage_list.append(neuron_voltage_list)

    # Show network membrane potential
    membrane_potential = np.array(neuron_voltage_list).reshape(image_height,image_width)
    pt.set_data(membrane_potential)
    plt.draw()
    plt.pause(1e-17)

    # Update states
    for j in range(0, image_height+2):
        for i in range(0, image_width+2):
            sensory_layer[j][i].update_state()

