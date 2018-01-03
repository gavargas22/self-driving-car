import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data_v2.npy')

train = train_data[:-850]
test = train_data[-850:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
Y_test = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, 
          validation_set=({'input': X_test}, {'targets': Y_test}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#Save the model
model.save(MODEL_NAME)


# tensorboard --logdir=foo:C:/Users/gavargas/Developer/self-driving-car/Driver/log


