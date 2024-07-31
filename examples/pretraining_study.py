# %% [markdown]
# # Pretraining Analysis

# %% [markdown]
# ## The Libraries

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import znnl as nl
import tensorflow_datasets as tfds
import numpy as np
from flax import linen as nn
import optax
from jax import random
from jax.lib import xla_bridge
import matplotlib.pyplot as plt
print(f"Using: {xla_bridge.get_backend().platform}")
import copy as cp
import jax.numpy as jnp
from papyrus.measurements import Loss, Accuracy, NTKEntropy, NTKTrace

# %%
# File path:

file_path = "/tikhome/kfritzler/PreTrainingStudy/Training_Records_2000epochs"

# %% [markdown]
# ## The Actual Data

# %%
data_generator = nl.data.MNISTGenerator(5000)

# %% [markdown]
# ## The Data Filter

# %%
def filter_numbers(data_set: dict, nr_type) -> dict:
    """
    Takes in a data set and returns a filtered data set only consisting of odd or even numbers.
    n; block dims: 128x1x1; grid dims

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

# %%
def select_subset_of_data(dataset: dict, seed: int, num_samples: int):
    """
    Selects a subset of data given an input dictionary.

    Arguments
    ---------
    data_set : dict
            data set
    seed : integer
            used to initialize a random number generator
    num_samples : integer
            Number of samples to be taken from the data set
    """

    # Generates random indices to select samples from the data:
    idx = random.randint(random.PRNGKey(seed), shape=(num_samples,), minval=0, maxval=dataset['targets'].shape[0])

    # Conerting into a NumPy array:
    idx = np.array(idx)

    # Filling a new dictionary:
    subset = {k: jnp.take(v, idx, axis=0) for k, v in dataset.items()}
    return subset

# %% [markdown]
# ## The Model

# %%
# Creating the model:

# Old model:
#class pretrained_CNN(nn.Module):
#    """
#    Simple CNN module.
#    """
#
#    @nn.compact
#    def __call__(self, x):
#        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
#        x = nn.relu(x)
#        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
#        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
#        x = nn.relu(x)
#        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
#        x = x.reshape((x.shape[0], -1))  # flatten
#        x = nn.Dense(features=300)(x)
#        x = nn.relu(x)
#        x = nn.Dense(10)(x)
#
#        return x
    
# New model:
class FullyConnectedNetwork(nn.Module):
    """
    Simple NN module.
    """

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)

        return x

# %% [markdown]
# ## The Pretraining (Even Numbers)

# %%
# We define the optimization algorithm here:
production_model = nl.models.FlaxModel(
            flax_module=FullyConnectedNetwork(),
            optimizer=optax.sgd(learning_rate=0.00005, momentum=0.9),
            input_shape=(1, 28, 28, 1),
            seed=0
        )

# To plot the losses, accuracy, etc. we put in recorders:

pretrain_recorder_even = nl.training_recording.JaxRecorder( 
    name="pretrain_recorder_even", 
    # where to save the data:
    storage_path=file_path, 
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()), 
    ],
    # number of samples to keep in memory before writing to disk:
    chunk_size=1e5, 
    # number of epochs between recording:
    update_rate=1
)

pretest_recorder_even = nl.training_recording.JaxRecorder( 
    name="pretest_recorder_even", 
    storage_path=file_path, 
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()), 
    ],
    chunk_size=1e5, 
    update_rate=1
)

pretest_recorder_odd = nl.training_recording.JaxRecorder( 
    name="pretest_recorder_odd", 
    storage_path=file_path, 
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()), 
    ],
    chunk_size=1e5, 
    update_rate=1
)

pretrain_recorder_even.instantiate_recorder(data_set=filter_numbers(data_generator.train_ds, 0), model=production_model)
pretest_recorder_even.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds, 0), model=production_model)
pretest_recorder_odd.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds, 1), model=production_model)

# The training strategy:
production_pretraining = nl.training_strategies.SimpleTraining(
    model=production_model, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[pretrain_recorder_even, pretest_recorder_even, pretest_recorder_odd]
)

# Here the training of the CNN takes place:
production_pretraining.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 0),
    test_ds=filter_numbers(data_generator.test_ds, 0),
    batch_size=64,
    epochs=2000,
)

# %%
# We gather the recorded losses and plot them over the epochs:
pretrain_report_even = pretrain_recorder_even.gather()
pretest_report_even = pretest_recorder_even.gather()
pretest_report_odd = pretest_recorder_odd.gather()

plt.plot(pretrain_report_even["loss"], 'o', mfc='None', label="Pretraining with even numbers")
plt.plot(pretest_report_even["loss"], '.', mfc="None", label="Testing with even numbers")
plt.plot(pretest_report_odd["loss"], '-', mfc="None", label="Testing with odd numbers")

plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Here we plot the accuracy:

plt.plot(pretrain_report_even["accuracy"], 'o', mfc='None', label="Pretraining with even numbers")
plt.plot(pretest_report_even["accuracy"], '.', mfc="None", label="Testing with even numbers")
plt.plot(pretest_report_odd["accuracy"], '-', mfc="None", label="Testing with odd numbers")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown]
# ## The Actual Training (Odd Numbers)

# %%
train_recorder_odd = nl.training_recording.JaxRecorder( 
    name="train_recorder_odd", 
    storage_path=file_path, 
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()), 
    ],
    chunk_size=1e5, 
    update_rate=1
)

test_recorder_odd = nl.training_recording.JaxRecorder( 
    name="test_recorder_odd", 
    storage_path=file_path, 
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()), 
    ],
    chunk_size=1e5, 
    update_rate=1
)

cv_recorder = nl.training_recording.JaxRecorder( 
    name="cv_recorder", 
    storage_path=file_path,
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
        NTKEntropy(name="ntk_entropy"), 
        NTKTrace(name="ntk_trace"),
    ],
    chunk_size=1e5,
    update_rate=1
)

ntk_computation = nl.analysis.JAXNTKComputation(
    apply_fn=production_model.ntk_apply_fn, 
    batch_size=10,
)

cv_recorder.instantiate_recorder(
    data_set=select_subset_of_data(dataset=filter_numbers(data_generator.test_ds, 1), seed=0, num_samples=100), 
    model=production_model, 
    ntk_computation=ntk_computation
)

train_recorder_odd.instantiate_recorder(data_set=filter_numbers(data_generator.train_ds, 1), model=production_model)
test_recorder_odd.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds, 1), model=production_model)


# The second training startegy:
production_training = nl.training_strategies.SimpleTraining(
    model=production_model, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[train_recorder_odd, test_recorder_odd, cv_recorder]
)

# Here the second training of the CNN takes place:
production_training.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 1),
    test_ds=filter_numbers(data_generator.test_ds, 1),
    batch_size=64,
    epochs=2000
)

# %%

# We gather the new recorded losses and plot them over the epochs:
train_report_odd = train_recorder_odd.gather()
test_report_odd = test_recorder_odd.gather()


plt.plot(train_report_odd["loss"], 'o', mfc='None', label="Training with odd numbers")
plt.plot(test_report_odd["loss"], 'o', mfc='None', label="Testing with odd numbers")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Here we plot the new acc1

# %%
cv_report = cv_recorder.gather()

fix, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)

axs[0, 0].plot(cv_report["ntk_trace"], 'o', mfc="None", label="Trace")
axs[0, 1].plot(cv_report["ntk_entropy"], 'o', mfc="None", label="Entropy")

axs[1, 0].plot(np.array(test_report_odd["loss"]), cv_report["ntk_trace"], 'o', mfc="None", label="Trace")
axs[1, 1].plot(np.array(test_report_odd["loss"]), cv_report["ntk_entropy"], 'o', mfc="None", label="Entropy")

axs[1, 0].xaxis.set_inverted(True)
axs[1, 1].xaxis.set_inverted(True)

axs[0, 0].set_yscale('log')
axs[1, 0].set_yscale('log')
axs[0, 0].set_xscale('log')
axs[0, 1].set_xscale('log')
axs[1, 0].set_xscale('log')
axs[1, 1].set_xscale('log')

axs[0, 0].set_xlabel("Epoch")
axs[0, 1].set_xlabel("Epoch")

axs[1, 0].set_xlabel("Test Loss")
axs[1, 1].set_xlabel("Test Loss")

axs[0, 0].set_ylabel("Trace")
axs[0, 1].set_ylabel("Entropy")
axs[1, 0].set_ylabel("Trace")
axs[1, 1].set_ylabel("Entropy")

axs[0, 0].legend()
plt.show()

# %% [markdown]
# ## NOT pretrained Model (Complete MNIST Data Set)

# %%
fresh_model = cp.deepcopy(FullyConnectedNetwork)


# We define the optimization algorithm here:
fresh_production_model = nl.models.FlaxModel(
            flax_module=fresh_model(),
            optimizer=optax.sgd(learning_rate=0.00005, momentum=0.9),
            input_shape=(1, 28, 28, 1),
            seed=0
        )

# To plot the losses, we put in a recorder:
fresh_train_recorder_odd = nl.training_recording.JaxRecorder( 
    name="fresh_train_recorder_odd", 
    storage_path=file_path,
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()), 
    ],
    chunk_size=1e5,
    update_rate=1
)

fresh_test_recorder_odd = nl.training_recording.JaxRecorder( 
    name="fresh_test_recorder_odd", 
    storage_path=file_path,
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
    ],
    chunk_size=1e5,
    update_rate=1
)


fresh_cv_recorder = nl.training_recording.JaxRecorder( 
    name="fresh_cv_recorder", 
    storage_path=file_path,
    measurements=[
        Loss(apply_fn=nl.loss_functions.CrossEntropyLoss()), 
        Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
        NTKEntropy(name="ntk_entropy"), 
        NTKTrace(name="ntk_trace"),
    ],
    chunk_size=1e5,
    update_rate=1
)

ntk_computation = nl.analysis.JAXNTKComputation(
    apply_fn=fresh_production_model.ntk_apply_fn, 
    batch_size=10,
)

fresh_cv_recorder.instantiate_recorder(
    data_set=select_subset_of_data(dataset=filter_numbers(data_generator.test_ds, 1), seed=0, num_samples=100), 
    model=fresh_production_model, 
    ntk_computation=ntk_computation
)

fresh_train_recorder_odd.instantiate_recorder(data_set=filter_numbers(data_generator.train_ds, 1), model=fresh_production_model)
fresh_test_recorder_odd.instantiate_recorder(data_set=filter_numbers(data_generator.test_ds, 1), model=fresh_production_model)



# The training startegy:
fresh_production_training = nl.training_strategies.SimpleTraining(
    model=fresh_production_model, 
    loss_fn=nl.loss_functions.CrossEntropyLoss(),
    accuracy_fn=nl.accuracy_functions.LabelAccuracy(), 
    recorders=[fresh_test_recorder_odd, fresh_train_recorder_odd, fresh_cv_recorder]
)

# Here the training of the CNN takes place:
fresh_production_training.train_model(
    train_ds=filter_numbers(data_generator.train_ds, 1),
    test_ds=filter_numbers(data_generator.test_ds,1),
    batch_size=64,
    epochs=2000,
)


# %%
# Plot the results for the random initialized model

# We gather the recorded losses and plot them over the epochs:
fresh_train_report_odd = fresh_train_recorder_odd.gather()
fresh_test_report_odd = fresh_test_recorder_odd.gather()

#cv_recorder4_report = .gather()

plt.plot(fresh_train_report_odd["loss"], 'o', mfc='None', label="Fresh training with odd numbers")
plt.plot(fresh_test_report_odd["loss"], '.', mfc="None", label="Fresh testing with odd numbers")
#plt.plot(cv_recorder4_report["ntk_entropy"], '-', mfc="None", label="Entropy")

plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Here we plot the new accuracy:
plt.plot(fresh_train_report_odd["accuracy"], 'o', mfc='None', label="Fresh training with odd numbers")
plt.plot(fresh_test_report_odd["accuracy"], '.', mfc="None", label="Fresh testing with odd numbers")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
fresh_cv_report = fresh_cv_recorder.gather()

plt.plot(fresh_cv_report["ntk_entropy"], '-', mfc="None", label="Entropy")
plt.plot(fresh_cv_report["ntk_trace"], 'o', mfc="None", label="Trace")


plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## The Plots

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

#plt.plot(train_report_odd["loss"], 'o', mfc='None', label="Training with odd numbers")

ax1.plot(fresh_test_report_odd["loss"], '.', mfc="None")
ax1.plot(test_report_odd["loss"], '.', mfc="None")
ax1.set_yscale("log")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Test Loss")
ax1.set_title("Loss vs. Epochs")

ax2.plot(fresh_test_report_odd["accuracy"], '.', mfc="None", label='fresh NN')
ax2.plot(test_report_odd["accuracy"], '.', mfc="None", label="pretrained NN")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Test Accuracy")
ax2.set_title("Accuracy vs. Epochs")
ax2.legend()


ax3.plot(fresh_test_report_odd["loss"], fresh_train_report_odd["loss"], '.', label='fresh NN Test-Loss over train-loss')
ax3.plot(test_report_odd["loss"], train_report_odd["loss"], '.', label='pretrained NN Test-Loss over train-loss')
ax3.set_yscale("log")
ax3.set_xlabel("Train Loss")
ax3.set_ylabel("Test Loss")
ax3.set_title("Test-Loss vs. Training-Loss")

ax4.plot(fresh_test_report_odd["accuracy"],fresh_train_report_odd["loss"], '.', label='fresh NN test-accuracy over train-loss')
ax4.plot(test_report_odd["accuracy"], train_report_odd["loss"], '.', label='pretrained NN test-accuracy over train-loss')
ax4.set_xlabel("Train Loss")
ax4.set_ylabel("Test Accuracy")
ax4.set_title("Test-Accuracy vs. Training-Loss")



# %%
cv_report = cv_recorder.gather()
fresh_cv_report = fresh_cv_recorder.gather()


fix, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)

axs[0, 0].plot(cv_report["ntk_trace"], 'o', mfc="None", label="pre-trained")
axs[0, 1].plot(cv_report["ntk_entropy"], 'o', mfc="None", label="pre-trained")
axs[0, 0].plot(fresh_cv_report["ntk_trace"], 'o', mfc="None", label="random")
axs[0, 1].plot(fresh_cv_report["ntk_entropy"], 'o', mfc="None", label="random")

axs[1, 0].plot(np.array(test_report_odd["loss"]), cv_report["ntk_trace"], 'o', mfc="None", label="pre-trained")
axs[1, 1].plot(np.array(test_report_odd["loss"]), cv_report["ntk_entropy"], 'o', mfc="None", label="pre-trained")
axs[1, 0].plot(np.array(test_report_odd["loss"]), fresh_cv_report["ntk_trace"], 'o', mfc="None", label="random")
axs[1, 1].plot(np.array(test_report_odd["loss"]), fresh_cv_report["ntk_entropy"], 'o', mfc="None", label="random")

axs[1, 0].xaxis.set_inverted(True)
axs[1, 1].xaxis.set_inverted(True)

axs[0, 0].set_yscale('log')
axs[1, 0].set_yscale('log')
axs[0, 0].set_xscale('log')
axs[0, 1].set_xscale('log')
axs[1, 0].set_xscale('log')
axs[1, 1].set_xscale('log')

axs[0, 0].set_xlabel("Epoch")
axs[0, 1].set_xlabel("Epoch")

axs[1, 0].set_xlabel("Test Loss")
axs[1, 1].set_xlabel("Test Loss")

axs[0, 0].set_ylabel("Trace")
axs[0, 1].set_ylabel("Entropy")
axs[1, 0].set_ylabel("Trace")
axs[1, 1].set_ylabel("Entropy")

axs[0, 0].legend()
plt.show()

# %%
pretrain_recorder_even.store()
pretest_recorder_even.store()
pretest_recorder_odd.store()
test_recorder_odd.store()
train_recorder_odd.store()
cv_recorder.store()
fresh_cv_recorder.store()
fresh_test_recorder_odd.store()
fresh_train_recorder_odd.store()


