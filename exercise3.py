# RECURSIVE NEURAL NETWORK REPOSITORY.
import numpy as np
np.random.seed(5)

from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K


# Preparation of the data.
names = open('files/dinosaurios.txt', 'r').read()
names = names.lower()

alphabet = list(set(names))
dim_data, dim_alphabet = len(names), len(alphabet)

car_a_ind = {car: ind for ind, car in enumerate(sorted(alphabet))}
ind_a_car = {ind: car for ind, car in enumerate(sorted(alphabet))}

# Creation of the Recursive Neural Network with Keras.
n_a = 25 # Number of units of the hidden layer.
input = Input(shape=(None, dim_alphabet))
a0 = Input(shape=(n_a,))

recursive_cell = SimpleRNN(n_a, activation='tanh', return_state=True)
output_layer = Dense(dim_alphabet, activation='softmax')

hs, _ = recursive_cell(input, initial_state=a0)
output = []
output.append(output_layer(hs))

model = Model([input, a0], output)
opt = SGD(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy')

# Training the Recursive Net.
with open('files/dinosaurios.txt') as f:
    examples = f.readline()
examples = [x.lower().strip() for x in examples]
np.random.shuffle(examples)

def train_generatior():
    while True:
        # Take an example randomly
        example = examples[np.random.randint(0,len(examples))]
        # Convert the example into numeric representation
        X = [None] + [car_a_ind[c] for c in example]
        # Y is X one character displace to right
        Y = X[1:] + [car_a_ind['\n']]
        # X and Y represented in onehot format
        x = np.zeros((len(X), 1, dim_alphabet))
        onehot = to_categorical(X[1:], dim_alphabet).reshape(len(X)-1, 1, dim_alphabet)
        x[1:,:,:] = onehot
        y= to_categorical(Y, dim_alphabet).reshape(len(X), dim_alphabet)

        # Initial start.
        a = np.zeros((len(X), n_a))

        yield [x, a], y


BATCH_SIZE = 80
NITS = 10000

for j in range(NITS):
    historial = model.fit_generator(train_generatior(), steps_per_epoch=BATCH_SIZE, epochs=1, verbose=0)

    # Print the evolution of the error for each 1000 iterations.
    if j%1000 == 0:
        print('\nIteration: %d, Error: %f' % (j, historial.history['loss'][0])+'\n')


def generate_name(model, car_a_num, dim_alphabel, n_a):
    # Initialising x and y with 0s
    x = np.zeros((1,1,dim_alphabel,))
    a = np.zeros((1, n_a))

    #Variable
    generated_name = ''
    fin_linea = '\n'
    car = -1

    cont = 0
    while(car != fin_linea and cont != 50):
        # generate predictions usinf RNN cell
        a, _ = recursive_cell(K.constant(x), initial_state=K.constant(a))
        y = output_layer(a)
        prediction = K.eval(y)

        # chose randomly an element of the prediction
        ix = np.random.choice(list(range(dim_alphabel)), p=prediction.ravel())

        # convert the chosen element to character and add to the generated name
        car = ind_a_car[ix]
        generated_name += car
        x = to_categorical(ix, dim_alphabel).reshape(1,1,dim_alphabel)
        a = K.eval(a)

        cont += 1

        if (cont == 50):
            generated_name += '\n'

    print(generated_name)

for i in range(10):
    generate_name(model, car_a_ind, dim_alphabet, n_a)
