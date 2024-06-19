# %% [markdown]
# # Pretraining Analysis

# %% [markdown]
# ## The Libraries

# %%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import znnl as nl
import tensorflow_datasets as tfds
import numpy as np
from flax import linen as nn
import optax
from jax.lib import xla_bridge
import matplotlib.pyplot as plt
print(f"Using: {xla_bridge.get_backend().platform}")
import copy as cp

# %% [markdown]
# ## The Actual Data

# %%
data_generator = nl.data.MNISTGenerator(2000)

# %% [markdown]
# ## The Data Filter

# %%
def filter_numbers(data_set: dict, nr_type) -> dict:
    """
    Takes in a data set and returns a filtered data set only consisting of odd or even numbers.

    Arguments
    ---------
    data_set : dict
            data set to be filtered.
    nr_type : integer
            For even numbers put 0, for odd numbers put 1

    Returns
    -------
    filtered_data_set : dict
            filtered data set
    """
    if nr_type == 1:
        
        # decompose data set into inputs and targets:
        inputs = data_set['inputs']
        targets = data_set['targets']
    
        # Get indices of odd numbers using targets:
        integer_targets = np.argmax(targets, axis=1)
        mod_targets = integer_targets % 2
        indices_of_odd_numbers = np.argwhere(mod_targets == 1).squeeze()
    
        # Take data according to indices: 
        inputs_odd = inputs[indices_of_odd_numbers]
        targets_odd = targets[indices_of_odd_numbers]
    
        # Construct and return new data set:
        return {"inputs": inputs_odd, "targets": targets_odd}
    
    if nr_type == 0:

        # decompose data set into inputs and targets:
        inputs = data_set['inputs']
        targets = data_set['targets']
    
        # Get indices of odd numbers using targets:
        integer_targets = np.argmax(targets, axis=1)
        mod_targets = integer_targets % 2
        indices_of_even_numbers = np.argwhere(mod_targets == 0).squeeze()
    
        # Take data according to indices: 
        inputs_even = inputs[indices_of_even_numbers]
        targets_even = targets[indices_of_even_numbers]
    
        # Construct and return new data set:
        return {"inputs": inputs_even, "targets": targets_even}
    
    else:
        return print("Please enter correct arguments. See documentation for help.")



# %% [markdown]
# ## The Pretraining (Even Numbers)

# %%
# The CNN model:
class pretrained_CNN(nn.Module):
    """
    Simple CNN module.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=300)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)

        return x

# We define the optimization algorithm here:
production_model1 = nl.models.FlaxModel(
            flax_module=pretrained_CNN(),
            optimizer=optax.sgd(learning_rate=0.00005, momentum=0.9),
            input_shape=(1, 28, 28, 1), # What input shape do I put in???
            seed=0
        )

# To plot the losses, we put in recorders:
train_recorder1 = nl.training_recording.JaxRecorder( 
    name="train_recorder1", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder_even = nl.training_recording.JaxRecorder( 
    name="test_recorder_even", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder_odd = nl.training_recording.JaxRecorder( 
    name="train_recorder_odd", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)

train_recorder1.instantiate_recorder(data_set=filter_numbers(data_generator.train_ds,0))
test_recorder_even.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds, 0))
test_recorder_odd.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds,1))


# The training startegy:
production_training1 = nl.training_strategies.SimpleTraining(
    model=production_model1, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[train_recorder1, test_recorder_even, test_recorder_odd]
)

# Here the training of the CNN takes place:
production_training1.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 0),
    test_ds=filter_numbers(data_generator.test_ds, 0),
    batch_size=64,
    epochs=1000
)

# We gather the recorded losses and plot them over the epochs:
train_report1 = train_recorder1.gather_recording()
test_report_even = test_recorder_even.gather_recording()
test_report_odd = test_recorder_odd.gather_recording()

plt.plot(train_report1.loss, 'o', mfc='None', label="Training 1: Pretraining with even numbers")
plt.plot(test_report_even.loss, '.', mfc="None", label="Test 1: Testing with even numbers")
plt.plot(test_report_odd.loss, '.', mfc="None", label="Test 2: Testing with odd numbers")

plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Here we plot the accuracy:
plt.plot(train_report1.accuracy, 'o', mfc='None', label="Training 1: Pretraining with even numbers")
plt.plot(test_report_even.accuracy, '.', mfc="None", label="Test 1: Testing with even numbers")
plt.plot(test_report_odd.accuracy, '.', mfc="None", label="Test 2: Testing with odd numbers")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#################


# %% [markdown]
# ## The Actual Training (Odd Numbers)

# %%

# We put in other recorders to plot the losses and accuracy of the second training:
train_recorder3 = nl.training_recording.JaxRecorder( 
    name="train_recorder3", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder3 = nl.training_recording.JaxRecorder( 
    name="test_recorder3", loss=True, accuracy=True, update_rate=1, chunk_size=1e5 )

train_recorder3.instantiate_recorder( data_set=filter_numbers(data_generator.train_ds,1) )
test_recorder3.instantiate_recorder( data_set=filter_numbers(data_generator.test_ds,1) )

# The second training startegy:
production_training3 = nl.training_strategies.SimpleTraining(
    model=production_model1, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[train_recorder3, test_recorder3]
)

# Here the second training of the CNN takes place:
production_training3.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 1),
    test_ds=filter_numbers(data_generator.test_ds, 1),
    batch_size=64,
    epochs=1000
)

# We gather the new recorded losses and plot them over the epochs:
train_report3 = train_recorder3.gather_recording()
test_report3 = test_recorder3.gather_recording()

plt.plot(train_report3.loss, 'o', mfc='None', label="Training 2: Actual training with odd numbers")
plt.plot(test_report3.loss, '.', mfc="None", label="Testing with odd numbers")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Here we plot the new accuracy:
plt.plot(train_report3.accuracy, 'o', mfc='None', label="Training 2: Actual training with odd numbers")
plt.plot(test_report3.accuracy, '.', mfc="None", label="Testing with odd numbers")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown]
# ## NOT pretrained Model (Complete MNIST Data Set)

# %%
CNN_4 = cp.deepcopy(pretrained_CNN)

# We define the optimization algorithm here:
production_model4 = nl.models.FlaxModel(
            flax_module=CNN_4(),
            optimizer=optax.sgd(learning_rate=0.00005, momentum=0.9),
            input_shape=(1, 28, 28, 1),
            seed=0
        )

# To plot the losses, we put in a recorder:
train_recorder4 = nl.training_recording.JaxRecorder( 
    name="train_recorder4", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder4 = nl.training_recording.JaxRecorder( 
    name="test_recorder_not4", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder_comp = nl.training_recording.JaxRecorder( 
    name="test_recorder_comp", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)

train_recorder4.instantiate_recorder(data_set=data_generator.train_ds)
test_recorder4.instantiate_recorder(data_set=data_generator.test_ds)
test_recorder_comp.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds, 1))

# The training startegy:
production_training_not_pretrained = nl.training_strategies.SimpleTraining(
    model=production_model4, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[train_recorder4, test_recorder4, test_recorder_comp]
)

# Here the training of the CNN takes place:
production_training_not_pretrained.train_model(
    train_ds=data_generator.train_ds,
    test_ds=data_generator.test_ds,
    batch_size=64,
    epochs=1000,
)

# We gather the recorded losses and plot them over the epochs:
train_report4 = train_recorder4.gather_recording()
test_report4 = test_recorder4.gather_recording()
test_report_comp = test_recorder_comp.gather_recording()

plt.plot(train_report4.loss, 'o', mfc='None', label="Training with complete data set")
plt.plot(test_report4.loss, '.', mfc="None", label="Testing with complete data set")
plt.plot(test_report_comp.loss, '-', mfc="None", label="Testing with odd numbers")

plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Here we plot the new accuracy:
plt.plot(train_report4.accuracy, 'o', mfc='None', label="Training with complete data set")
plt.plot(test_report4.accuracy, '.', mfc="None", label="Testing with complete data set")
plt.plot(test_report_comp.accuracy, '-', mfc="None", label="Testing with odd numbers")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown]
# ## The Plots

# %%
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# test_report_comp = NOT pretrained model, tested with odd numbers

# test_report3 = pretrained model, tested with odd numbers

ax1.plot(test_report_comp.loss, '.', mfc="None", label='NN Test')
ax1.plot(test_report3.loss, '.', mfc="None", label="pretrained NN Test")
ax1.set_yscale("log")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Test Loss")


ax2.plot(test_report_comp.accuracy, '.', mfc="None", label='fresh NN')
ax2.plot(test_report3.accuracy, '.', mfc="None", label="pretrained NN")
ax2.set_yscale("log")
ax2.legend()
ax2.set_xlabel("Epochs")
ax1.set_ylabel("Test Accuracy")

ax3.plot(test_report_comp.loss, train_report4.loss, '.', label='NN Test over Train')
ax3.plot(test_report3.loss, train_report3.loss, '.', label='pretrained NN Test over Train')
ax3.set_yscale("log")
ax3.set_xlabel("Train Loss")
ax3.set_xlabel("Test Loss")

ax4.plot(test_report_comp.accuracy, train_report4.loss, '.', label='NN Test over Train')
ax4.plot(test_report3.accuracy, train_report3.loss, '.', label='pretrained NN Test over Train')
ax4.set_yscale("log")
ax4.set_xlabel("Train Loss")
ax4.set_ylabel("Test Accuracy")



