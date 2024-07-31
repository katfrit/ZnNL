# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
def calculate_average_per_epoch(file_paths, dataset_name, error):
    """
    Calculate the mean per epoch of specific values from multiple HDF5 files.
    Parameters:
    - file_paths: List of strings, paths to the HDF5 files.
    - dataset_name: String, the name of the dataset to extract values from.
    - nepochs: Integer, the number of epochs.

    Returns:
    - average_values_per_epoch: Numpy array, average values per epoch.
    """
    all_runs_values = []

    for file_path in file_paths:
        with h5py.File(file_path, "r") as f:
            values = f[dataset_name][:]
            all_runs_values.append(values)
    
    # Calculate the average values per epoch
    average_values_per_epoch = np.mean(all_runs_values, axis=0)

    # Calculate the standard deviation:
    standard_deviation = np.std(all_runs_values, axis=0)
    
    if error == 0:
        return average_values_per_epoch
    if error == 1:
        return standard_deviation

# %% [markdown]
# ### Collective Variables

# %%
## List of HDF5 file paths

# The collective variables from the pretrained model:
cv_file_paths = [
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run1.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run2.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run3.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run4.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run5.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run6.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run7.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run8.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run9.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/cv_recorder_run10.h5"
]

# The collective variables from the fresh model:
fresh_cv_file_paths = [
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run1.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run2.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run3.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run4.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run5.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run6.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run7.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run8.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run9.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_cv_recorder_run10.h5"
]

# The losses and acuarcies from the pretrained model:
test_file_paths = [
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run1.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run2.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run3.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run4.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run5.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run6.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run7.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run8.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run9.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/test_recorder_odd_run10.h5"
]



## Define the number of epochs and the dataset name

nepochs = 700

# Data from actual training of pretrained model:
train_trace = 'ntk_trace'
train_entropy = 'ntk_entropy'
train_loss = 'loss'

# Data from training of fresh model:
fresh_trace = 'ntk_trace'
fresh_entropy = 'ntk_entropy'

# %%
## Calculate the average values per epoch

# pretrained model:
average_train_trace_per_epoch = calculate_average_per_epoch(cv_file_paths, train_trace, 0)
average_train_entropy_per_epoch = calculate_average_per_epoch(cv_file_paths, train_entropy, 0)
average_test_loss_per_epoch = calculate_average_per_epoch(test_file_paths, train_loss, 0)

# fresh model:
average_fresh_trace_per_epoch = calculate_average_per_epoch(fresh_cv_file_paths, fresh_trace, 0)
average_fresh_entropy_per_epoch = calculate_average_per_epoch(fresh_cv_file_paths, fresh_entropy, 0)

# number of epochs:
y_values = range(nepochs)

# %%
# Errorbars:
#n = 10

#error_train_trace = np.array(calculate_average_per_epoch(cv_file_paths, train_trace, 1)/(np.sqrt(n)))
#error_train_entropy = np.array(calculate_average_per_epoch(cv_file_paths, train_entropy, 1)/(np.sqrt(n)))
#error_fresh_trace = np.array(calculate_average_per_epoch(fresh_cv_file_paths, train_trace, 1)/(np.sqrt(n)))
#error_fresh_entropy = np.array(calculate_average_per_epoch(fresh_cv_file_paths, train_entropy, 1)/(np.sqrt(n)))

# %%
fix, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)

fix.suptitle('Average Values per Epochs', fontsize=16)

axs[0, 0].plot(y_values, average_train_trace_per_epoch, 'o', mfc="None", label="pre-trained")
#axs[0, 0].fill_between(y_values, average_train_trace_per_epoch - error_train_trace, average_train_trace_per_epoch + error_train_trace, alpha=0.2, label='error band')
axs[0, 1].plot(y_values, average_train_entropy_per_epoch, 'o',  mfc="None", label="pre-trained")
axs[0, 0].plot(y_values, average_fresh_trace_per_epoch, 'o', mfc="None", label="fresh")
axs[0, 1].plot(y_values, average_fresh_entropy_per_epoch, 'o', mfc="None", label="fresh")

axs[1, 0].plot(np.array(average_test_loss_per_epoch), average_train_trace_per_epoch, 'o', mfc="None", label="pre-trained")
axs[1, 1].plot(np.array(average_test_loss_per_epoch), average_train_entropy_per_epoch, 'o', mfc="None", label="pre-trained")
axs[1, 0].plot(np.array(average_test_loss_per_epoch), average_fresh_trace_per_epoch, 'o', mfc="None", label="fresh")
axs[1, 1].plot(np.array(average_test_loss_per_epoch), average_fresh_entropy_per_epoch, 'o', mfc="None", label="fresh")

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

plt.savefig('/tikhome/kfritzler/PreTrainingStudy/Pictures')

# %% [markdown]
# ### Losses and Accuracies

# %%
## List of HDF5 file paths

# The losses and acuarcies from the pretrained model:
train_file_paths = [
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run1.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run2.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run3.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run4.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run5.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run6.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run7.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run8.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run9.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run10.h5"
]

# The losses and acuarcies from the fresh model:
fresh_file_paths = [
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run1.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run2.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run3.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run4.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run5.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run6.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run7.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run8.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run9.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/fresh_train_recorder_odd_run10.h5"
]

# The losses and acuarcies from the pretrained model:
train_file_paths = [
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run1.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run2.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run3.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run4.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run5.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run6.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run7.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run8.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run9.h5",
    "/tikhome/kfritzler/PreTrainingStudy/Training_Records/train_recorder_odd_run10.h5"
]


## Define the number of epochs and the dataset name

nepochs = 700

# Data from actual training of pretrained model:
train_accuray = 'accuracy'
train_loss = 'loss'

# Data from training of fresh model:
fresh_accuracy = 'accuracy'
fresh_loss = 'loss'

# %%
## Calculate the average values per epoch

# pretrained model:
average_train_accuracy_per_epoch = calculate_average_per_epoch(train_file_paths, train_accuray, 0)
average_train_loss_per_epoch = calculate_average_per_epoch(train_file_paths, train_loss, 0)

# fresh model:
average_fresh_accuracy_per_epoch = calculate_average_per_epoch(fresh_file_paths, fresh_accuracy, 0)
average_fresh_loss_per_epoch = calculate_average_per_epoch(fresh_file_paths, fresh_loss, 0)

# number of epochs:
y_values = range(nepochs)

# %%
fix, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)

fix.suptitle('Average Values per Epochs', fontsize=16)

axs[0, 0].plot(y_values, average_train_loss_per_epoch, 'o', mfc="None", label="pre-trained")
axs[0, 1].plot(y_values, average_train_accuracy_per_epoch, 'o',  mfc="None", label="pre-trained")
axs[0, 0].plot(y_values, average_fresh_loss_per_epoch, 'o', mfc="None", label="fresh")
axs[0, 1].plot(y_values, average_fresh_accuracy_per_epoch, 'o', mfc="None", label="fresh")

axs[1, 0].plot(np.array(average_train_loss_per_epoch), average_train_loss_per_epoch, 'o', mfc="None", label="pre-trained")
axs[1, 1].plot(np.array(average_train_loss_per_epoch), average_train_accuracy_per_epoch, 'o', mfc="None", label="pre-trained")
axs[1, 0].plot(np.array(average_train_loss_per_epoch), average_fresh_loss_per_epoch, 'o', mfc="None", label="fresh")
axs[1, 1].plot(np.array(average_train_loss_per_epoch), average_fresh_accuracy_per_epoch, 'o', mfc="None", label="fresh")

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

axs[1, 0].set_xlabel("Train Loss")
axs[1, 1].set_xlabel("Train Loss")

axs[0, 0].set_ylabel("Loss")
axs[0, 1].set_ylabel("Accuracy")
axs[1, 0].set_ylabel("Loss")
axs[1, 1].set_ylabel("Accuracy")

axs[0, 0].legend()

plt.show()

# %% [markdown]
# # TO DO:
# - add error bars with np.fill_between
# - run with 2000-epoch files


