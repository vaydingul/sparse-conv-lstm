
import collections
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from itertools import islice, tee

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings
warnings.filterwarnings("ignore")



def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def window(iterable, n=2):
    "s -> (s0, ...,s(n-1)), (s1, ...,sn), (s2, ..., s(n+1)), ..."
    iters = tee(iterable, n)
    # Could use enumerate(islice(iters, 1, None), 1) to avoid consume(it, 0), but that's
    # slower for larger window sizes, while saving only small fixed "noop" cost
    for i, it in enumerate(iters):
        consume(it, i)
    return zip(*iters)

def main(args):
    # Device is CPU since it is a local machine
    pytorch_device = torch.device('cpu')

    # It is the input given to the main script:
    # the directory of the config.yaml
    config_path = args.config_path

    # Load the configs
    configs = load_config_data(config_path)

    # The ´dataset_params´ section of the config file
    dataset_config = configs['dataset_params']

    # The ´train_data_loader´ section of the config file
    train_dataloader_config = configs['train_data_loader']

    # The ´val_data_loader´ section of the config file
    val_dataloader_config = configs['val_data_loader']

    # Validation set batch size
    val_batch_size = val_dataloader_config['batch_size']

    # Train set batch size
    train_batch_size = train_dataloader_config['batch_size']

    # The ´model_params´ section of the config file
    model_config = configs['model_params']

    # The ´train_params´ section of the config file
    train_hypers = configs['train_params']

    # Grid size of each voxel
    grid_size = model_config['output_shape']
    # Number of classes/categories
    num_class = model_config['num_class']
    # Whether the labels will be ignored or not
    ignore_label = dataset_config['ignore_label']

    # The pretrained model loading directory
    model_load_path = train_hypers['model_load_path']
    # The pretrained model saving directory
    model_save_path = train_hypers['model_save_path']

    # The directory of the labeling config file
    SemKITTI_label_name = get_SemKITTI_label_name(
        dataset_config["label_mapping"])

    # Integer corresponding to the unique labels
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    # String representation of the labels
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    # Build the initial model
    my_model = model_builder.build(model_config)

    # If there is a predefined one in the loading path,
    # then fetch it.
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    # Transmit to the current device
    my_model.to(pytorch_device)

    # Construct the optimizer settings
    optimizer = optim.Adam(my_model.parameters(),
                           # Learning rate
                           lr=train_hypers["learning_rate"])

    # Loss function configuration
    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    # Dataset builder, both train and validation set.
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    
    # Epoch count initialization
    epoch = 0
    
    # Best validation mean intersection over union initialization
    best_val_miou = 0
    # Set the training mode for the model!
    my_model.train()

    # ==  Some dummy variables for verbosing ==
    # Total number of iterations
    global_iter = 0
    # Checking period over iterations
    check_iter = train_hypers['eval_every_n_steps']
    # The maximum number of epoch that model  will be trained
    MAXIMUM_NUMBER_OF_EPOCHS = train_hypers['max_num_epochs']
    
    SEQUENCE_LENGTH = 3
    # ====

    while epoch < MAXIMUM_NUMBER_OF_EPOCHS:

        # Initialize the list of losses
        loss_list = []
        # Set the tqdm bar
        pbar = tqdm(total=len(train_dataset_loader))

        # ? lr_scheduler.step(epoch)

        #for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
        for i_iter, train_data in enumerate(window(train_dataset_loader, SEQUENCE_LENGTH)):

        

            if global_iter % check_iter == 0: # and epoch >= 1:

                #* ############  VALIDATION SET validation ##########################
                # Set the evaluation/inference mode for the model!
                my_model.eval()
                # Initialize an empty list to store
                # histograms of predictions
                hist_list = []
                # Initialize an empty list to store
                # loss values
                val_loss_list = []

                # Do not calculate gradients during inference
                with torch.no_grad():
                    # Iterate over validation set
                    #for val_iter_no, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                    #        val_dataset_loader):

                    for val_data in window(val_dataset_loader, SEQUENCE_LENGTH):
                        print("==============")
                        """
                        # TODO: Do that in a collate function
                        # Convert the val_pt_fea to Torch Float Tensor
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]

                        # Convert the val_grid to Torch Tensor
                        val_grid_ten = [torch.from_numpy(i).to(
                            pytorch_device) for i in val_grid]

                        # Convert the val_vox_label to Torch Tensor
                        val_label_tensor = val_vox_label.type(
                            torch.LongTensor).to(pytorch_device)
                        """
                        
                        val_pt_fea_ten = []
                        val_grid_ten = []
                        val_grid = []
                        val_pt_labs = []
                        for datum in val_data:
                            
                            val_pt_fea_ten.append(torch.from_numpy(datum[-1][0]).type(torch.FloatTensor).to(pytorch_device))
                            val_grid_ten.append(torch.from_numpy(datum[2][0]).to(pytorch_device))
                            
                            val_grid.append(datum[2][0])
                            val_pt_labs.append(datum[3][0])


                        #val_label_tensor = torch.stack([datum[1] for datum in data])
                        val_label_tensor = val_data[int((SEQUENCE_LENGTH+1)*0.5) - 1][1].type(
                            torch.LongTensor).to(pytorch_device)



                        #! Execute the model!
                        predict_labels = my_model(
                            val_pt_fea_ten, val_grid_ten, SEQUENCE_LENGTH)

                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)

                        #! Calculate the loss ==> lovasz_softmax + loss_func
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) +\
                            loss_func(predict_labels.detach(),
                                      val_label_tensor)

                        # Make predictions based on the label having the highest probability
                        predict_labels = torch.argmax(predict_labels, dim=1)

                        # Transmit them to CPU
                        predict_labels = predict_labels.cpu().detach().numpy()

                        val_grid = [val_grid[int((SEQUENCE_LENGTH+1)*0.5) - 1]]
                        val_pt_labs = [val_pt_labs[int((SEQUENCE_LENGTH+1)*0.5) - 1]]
                        # For every point coordinate in point cloud
                        for count, i_val_grid in enumerate(val_grid):

                            # Calculate the square histogram of the predictions
                            # and store in the hist_list list
                            hist_list.append(fast_hist_crop(predict_labels[
                                count, val_grid[count][:,
                                                       0], val_grid[count][:, 1],
                                val_grid[count][:, 2]], val_pt_labs[count],
                                unique_label))

                        # Store the calculated loss in val_loss_list list
                        val_loss_list.append(loss.detach().cpu().numpy())
                
                # Switch to the training mode
                # TODO: Check if it can be removed from here?
                my_model.train()
                
                # Calculate the per class intersection over union
                iou = per_class_iu(sum(hist_list))

                print('Validation per class iou: ')
                # For each label
                for class_name, class_iou in zip(unique_label_str, iou):

                    # Print out the results of the intersection over union
                    print('%s : %.2f%%' % (class_name, class_iou * 100))

                # Calculate the mean intersection over union
                val_miou = np.nanmean(iou) * 100
                del val_grid, val_grid_ten

                # If there is an improvement on the 
                # mean intersection over union, 
                # then save the model
                if best_val_miou < val_miou:
                    
                    # Basic magnitude check and iteration
                    best_val_miou = val_miou

                    # Save model
                    torch.save(my_model.state_dict(), model_save_path)

                # Print out the current and best value of 
                # mean intersection over union
                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                
                # Print out the mean loss value across the batch!
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))
                #* #############################################################################

            #* ##################################### TRAINING ROUTINE ##########################
            
            """
            # Convert the train_pt_fea to Torch Float Tensor
            train_pt_fea_ten = [torch.from_numpy(i).type(
                torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]

            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]

            # Convert the train_grid to Torch  Tensor
            train_vox_ten = [torch.from_numpy(i).to(
                pytorch_device) for i in train_grid]

            # Convert the train_vox_label to Torch  Tensor
            point_label_tensor = train_vox_label.type(
                torch.LongTensor).to(pytorch_device)
            """

            train_pt_fea_ten = []
            train_grid_ten = []
            train_vox_ten = []
            train_grid = []

            for datum in train_data:
                
                train_pt_fea_ten.append(torch.from_numpy(datum[-1][0]).type(torch.FloatTensor).to(pytorch_device))
                train_vox_ten.append(torch.from_numpy(datum[2][0]).to(pytorch_device))
                train_grid_ten.append(torch.from_numpy(datum[2][0][:, :2]).to(pytorch_device))
                

                train_grid.append(datum[2][0])


            #val_label_tensor = torch.stack([datum[1] for datum in data])
            point_label_tensor = train_data[int((SEQUENCE_LENGTH+1)*0.5) - 1][1].type(
                torch.LongTensor).to(pytorch_device)

            #! Forward + Backward + Optimize

            #! Execute the model
            outputs = my_model(
                train_pt_fea_ten, train_vox_ten, SEQUENCE_LENGTH)
            #! Calculate the loss
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(
                outputs, point_label_tensor)
            #! Calculate the backward pass
            loss.backward()
            #! Optimize one step
            optimizer.step()

            # Store the calculated loss in the loss_list list
            loss_list.append(loss.item())
            
            # If it is 1000th iteration
            if global_iter % 1000 == 0:
                # and if the loss_list is not empty
                if len(loss_list) > 0:
                    # Then, print out:
                    # - Epoch number
                    # - Iteration number
                    # - Mean training loss over iterations
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            # Cancel out the backward computation
            optimizer.zero_grad()

            # Increment the progress bar
            pbar.update(1)

            # Increment the iteration
            global_iter += 1

            # If the iteration count equals to some predetermined number
            if global_iter % check_iter == 0:

                # And if the loss_list is not empty
                if len(loss_list) > 0:
                    # Then, print out:
                    # - Epoch number
                    # - Iteration number
                    # - Mean training loss over iterations
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

        # After one epoch of training and validation, clear the progress bar
        pbar.close()
        # Increment the epoch count
        epoch += 1

        #* #############################################################################



if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path',
                        default='config/semantickitti_cpu.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
