import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import math

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

        self.inhibit_input_list = []
        self.inhibit_weight_list = []

        self.fired = False
        self.fire_time = -1000
        self.presynaptic_fire_list = []

        self.alpha = 1
        self.thao = 20

    def register_excitatory_input(self, input_neuron, min_value, max_value):
        self.input_list.append(input_neuron)
        self.weight_list.append((max_value - min_value) * np.random.rand(1)[0] + min_value)

    def register_inhibitory_input(self, input_neuron, min_value, max_value):
        self.inhibit_input_list.append(input_neuron)
        self.inhibit_weight_list.append(-((max_value - min_value) * np.random.rand(1)[0] + min_value))

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

        for i in range(len(self.inhibit_input_list)):
            presynaptic_state = self.inhibit_input_list[i].get_fired()
            if presynaptic_state:
                total_input += self.inhibit_weight_list[i]

        self.next_v += 0.04 * self.next_v ** 2 + 5 * self.next_v + 140 - self.next_u + total_input
        self.next_u += self.a * (self.b * self.next_v - self.next_u)

        if self.fired:
            self.next_v = self.c
            self.next_u = self.u + self.d

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

class IzhikevichInhibitoryNeuron:
    def __init__(self):
        self.a = 0.02 + 0.08 * np.random.rand(1)[0]
        self.b = 0.25 - 0.05 * np.random.rand(1)[0]
        self.c = -65
        self.d = 2

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

    def register_excitatory_input(self, input_neuron, min_value, max_value):
        self.input_list.append(input_neuron)
        self.weight_list.append((max_value - min_value) * np.random.rand(1)[0] + min_value)

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

        self.next_v += 0.04 * self.next_v ** 2 + 5 * self.next_v + 140 - self.next_u + total_input
        self.next_u += self.a * (self.b * self.next_v - self.next_u)

        if self.fired:
            self.next_v = self.c
            self.next_u = self.u + self.d

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

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(full_train_x, full_train_y), (test_x, test_y) = mnist.load_data()
input_image = full_train_x[1, :, :]
input_image = (input_image.astype('float32') / 255) * 5

# Input parameters
time = 500
image_height = 28
image_width = 28

# Define layers
sensory_layer = []
for j in range(3*image_height+2):
    row = []
    for i in range(3*image_width+2):
        row.append(IzhikevichExitatoryNeuron())
    sensory_layer.append(row)

feature_layer_excitatory = []
for j in range(3 * image_height + 2):
    row = []
    for i in range(3 * image_width + 2):
        row.append(IzhikevichExitatoryNeuron())
    feature_layer_excitatory.append(row)

feature_layer_inhibitory = []
for j in range(image_height):
    row = []
    for i in range(image_width):
        row.append(IzhikevichInhibitoryNeuron())
    feature_layer_inhibitory.append(row)

output_layer = []
for j in range(image_height):
    row = []
    for i in range(image_width):
        row.append(IzhikevichExitatoryNeuron())
    output_layer.append(row)

# Conect layers
e_synapse_min = 10
e_synapse_max = 15
for j in range(1,3*image_height+1):
    for i in range(1,3*image_width+1):
        a=1
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j-1][i-1], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j-1][i], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j-1][i+1], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j][i-1], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j][i+1], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j+1][i-1], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j+1][i], e_synapse_min, e_synapse_max)
        sensory_layer[j][i].register_excitatory_input(sensory_layer[j+1][i+1], e_synapse_min, e_synapse_max)

e_synapse_min = 10
e_synapse_max = 15
i_synapse_min = 20
i_synapse_max = 25
for j in range(1, 3*image_height + 1):
    for i in range(1, 3*image_width + 1):
        feature_layer_excitatory[j][i].register_excitatory_input(sensory_layer[j][i], e_synapse_min, e_synapse_max)

        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j - 1][i - 1], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j - 1][i], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j - 1][i + 1], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j][i - 1], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j][i + 1], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j + 1][i - 1], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j + 1][i], e_synapse_min, e_synapse_max)
        feature_layer_excitatory[j][i].register_excitatory_input(feature_layer_excitatory[j + 1][i + 1], e_synapse_min, e_synapse_max)

        feature_layer_excitatory[j][i].register_inhibitory_input(feature_layer_inhibitory[int((j-1)/3)][int((i-1)/3)], i_synapse_min, i_synapse_max)


e_synapse_min = 10
e_synapse_max = 15
for j in range(image_height):
    exc_j = j * 3 + 1
    for i in range(image_width):
        exc_i = i * 3 + 1
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j][exc_i], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j][exc_i+1], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j][exc_i+2], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j+1][exc_i], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j+1][exc_i+1], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j+1][exc_i+2], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j+2][exc_i], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j+2][exc_i+1], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(sensory_layer[exc_j+2][exc_i+2], e_synapse_min, e_synapse_max)

        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j][exc_i], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j][exc_i + 1], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j][exc_i + 2], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j + 1][exc_i], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j + 1][exc_i + 1], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j + 1][exc_i + 2], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j + 2][exc_i], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j + 2][exc_i + 1], e_synapse_min, e_synapse_max)
        feature_layer_inhibitory[j][i].register_excitatory_input(feature_layer_excitatory[exc_j + 2][exc_i + 2], e_synapse_min, e_synapse_max)


e_synapse_min = 10
e_synapse_max = 15

np.random.seed(123)
random_indexes = np.arange(9*image_height*image_width)
np.random.shuffle(random_indexes)

index = 0
for j in range(image_height):
    for i in range(image_width):
        for k in range(9):
            val_j = int(random_indexes[index] / (3*image_width)) + 1
            val_i = random_indexes[index] % (3*image_width) + 1
            output_layer[j][i].register_excitatory_input(feature_layer_excitatory[val_j][val_i], e_synapse_min, e_synapse_max)
            index += 1

# Prepare plotting window
pt = plt.imshow(-65 * np.ones(3*image_height * 3*image_width).reshape(3*image_height, 3*image_width), vmin=-100, vmax=30)
plt.colorbar(orientation='vertical')

# Run network
sensory_layer_potential = []
excitatory_feature_potential= []
inhibitory_feature_potential= []
output_layer_potential = []

for t in range(time):
    # Process input
    sensory_neuron = []
    for j in range(1, 3*image_height+1):
        for i in range(1, 3*image_width+1):
            input_j = int((j - 1) / 3)
            input_i = int((i - 1) / 3)
            sensory_layer[j][i].evaluate_state(input_image[input_j, input_i])
            sensory_layer[j][i].evaluate_synaptic_rule()

            sensory_neuron.append(sensory_layer[j][i].get_voltage())
    sensory_layer_potential.append(sensory_neuron)

    excitatory_feature_neuron = []
    for j in range(1, 3 * image_height + 1):
        for i in range(1, 3 * image_width + 1):
            feature_layer_excitatory[j][i].evaluate_state(0 * np.random.normal(0, 1))
            feature_layer_excitatory[j][i].evaluate_synaptic_rule()

            excitatory_feature_neuron.append(feature_layer_excitatory[j][i].get_voltage())
    excitatory_feature_potential.append(excitatory_feature_neuron)

    inhibitory_feature_neuron = []
    for j in range(image_height):
        for i in range(image_width):
            feature_layer_inhibitory[j][i].evaluate_state(0)
            # sensory_layer[j][i].evaluate_synaptic_rule()

            inhibitory_feature_neuron.append(feature_layer_inhibitory[j][i].get_voltage())
    inhibitory_feature_potential.append(inhibitory_feature_neuron)

    output_neuron = []
    for j in range(image_height):
        for i in range(image_width):
            output_layer[j][i].evaluate_state(0)
            output_layer[j][i].evaluate_synaptic_rule()

            output_neuron.append(output_layer[j][i].get_voltage())
    output_layer_potential.append(output_neuron)

    # Update states
    for j in range(0, 3*image_height+2):
        for i in range(0, 3*image_width+2):
            sensory_layer[j][i].update_state()

    for j in range(0, 3*image_height+2):
        for i in range(0, 3*image_width+2):
            feature_layer_excitatory[j][i].update_state()

    for j in range(image_height):
        for i in range(image_width):
            feature_layer_inhibitory[j][i].update_state()

    for j in range(image_height):
        for i in range(image_width):
            output_layer[j][i].update_state()

    # Display membrane potential
    vol = np.array(excitatory_feature_neuron).reshape(3*image_height, 3*image_width)
    pt.set_data(vol)
    plt.draw()
    plt.pause(1e-17)

    # Display membrane potential
    # vol = np.array(excitatory_feature_neuron).reshape(3 * image_height, 3 * image_width)
    # pt.set_data(vol)
    # plt.draw()
    # plt.pause(1e-17)

    # Display membrane potential
    # vol = np.array(inhibitory_feature_neuron).reshape(image_height, image_width)
    # pt.set_data(vol)
    # plt.draw()
    # plt.pause(1e-17)

    # Display membrane potential
    # vol = np.array(output_neuron).reshape(image_height, image_width)
    # pt.set_data(vol)
    # plt.draw()
    # plt.pause(1e-17)

tv = range(time)
ax1 = plt.subplot(411)
plt.plot(tv, np.array(sensory_layer_potential).reshape(time, 3*image_height, 3*image_height)[:,30,30])
plt.ylim(-100, 30)

ax2 = plt.subplot(412)
plt.plot(tv, np.array(excitatory_feature_potential).reshape(time, 3*image_height, 3*image_height)[:,30,30])
plt.ylim(-100, 30)

ax3 = plt.subplot(413)
plt.plot(tv, np.array(inhibitory_feature_potential).reshape(time, image_height, image_height)[:,3,3])
plt.ylim(-100, 30)

ax4 = plt.subplot(414)
plt.plot(tv, np.array(output_layer_potential).reshape(time, image_height, image_height)[:,3,3])
plt.ylim(-100, 30)

plt.show()