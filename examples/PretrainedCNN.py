# %% [markdown]
# ## Pretraining a CNN with even numbers, then training it with odd numbers

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import znnl as nl
import tensorflow_datasets as tfds
import numpy as np
from flax import linen as nn
import optax
from jax.lib import xla_bridge
import matplotlib.pyplot as plt
print(f"Using: {xla_bridge.get_backend().platform}")

# %% [markdown]
# ### Getting the Data

# %%
data_generator = nl.data.MNISTGenerator()

# %% [markdown]
# ### Filtering the Data

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
# ### Define the agent

# %%
# The CNN model:
class CNN(nn.Module):
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

# %%
# We define the optimization algorithm here:
production_model = nl.models.FlaxModel(
            flax_module=CNN(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 28, 28, 1), # What input shape do I put in???
            seed=0
        )

# %%
# To plot the losses, we put in a recorder:
train_recorder = nl.training_recording.JaxRecorder( 
    name="train_recorder", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder = nl.training_recording.JaxRecorder( 
    name="test_recorder", loss=True, accuracy=True, update_rate=1, chunk_size=1e5 )

train_recorder.instantiate_recorder( data_set=filter_numbers(data_generator.train_ds,0) )
test_recorder.instantiate_recorder( data_set=filter_numbers(data_generator.test_ds, 0) )

# %%
# We put in other recorders to plot the losses and accuracy of the second training:
train_recorder_2 = nl.training_recording.JaxRecorder( 
    name="train_recorder_2", loss=True, accuracy=True, update_rate=1, chunk_size=1e5)
test_recorder_2 = nl.training_recording.JaxRecorder( 
    name="test_recorder_2", loss=True, accuracy=True, update_rate=1, chunk_size=1e5 )

train_recorder_2.instantiate_recorder( data_set=filter_numbers(data_generator.train_ds,1) )
test_recorder_2.instantiate_recorder( data_set=filter_numbers(data_generator.test_ds, 1) )

# %%
# The training startegy:
production_training = nl.training_strategies.SimpleTraining(
    model=production_model, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[train_recorder, test_recorder]
)

# %%
# The second training startegy:
production_training_2 = nl.training_strategies.SimpleTraining(
    model=production_model, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[train_recorder_2, test_recorder_2]
)

# %%
# Here the training of the CNN takes place:
production_training.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 0),
    test_ds=filter_numbers(data_generator.test_ds, 0),
    batch_size=64,
    epochs=150
)

# %%
# Here the second training of the CNN takes place:
production_training_2.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 1),
    test_ds=filter_numbers(data_generator.test_ds, 1),
    batch_size=64,
    epochs=150
)

# %%
# We gather the recorded losses and plot them over the epochs:
train_report = train_recorder.gather_recording()
test_report = test_recorder.gather_recording()

plt.plot(train_report.loss, 'o', mfc='None', label="Train")
plt.plot(test_report.loss, '.', mfc="None", label="Test")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# We gather the new recorded losses and plot them over the epochs:
train_report_2 = train_recorder_2.gather_recording()
test_report_2 = test_recorder_2.gather_recording()

plt.plot(train_report_2.loss, 'o', mfc='None', label="Train")
plt.plot(test_report_2.loss, '.', mfc="None", label="Test")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# Here we plot the accuracy:
plt.plot(train_report.accuracy, 'o', mfc='None', label="Train")
plt.plot(test_report.accuracy, '.', mfc="None", label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
# Here we plot the new accuracy:
plt.plot(train_report_2.accuracy, 'o', mfc='None', label="Train")
plt.plot(test_report_2.accuracy, '.', mfc="None", label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%



