from tqdm.auto import tqdm
import math
import torch
from update_utilities import update_utilities_class
import os
import numpy as np
import time
import copy

class train_test_loop_class:
    def __init__(self, model:torch.nn.Module, 
                 train_loader:torch.utils.data.DataLoader,
                 val_loader:torch.utils.data.DataLoader,
                 test_loader, epochs, print_every_n_batch,
                 device,model_name, optimizer,calculate_accuracy,problem_type,overwrite_message,update_loss_fn=False,
                 print_result = True, print_full = True, lr_rate_tuning=False, 
                 clip_batch=False,clip_batch_size=32, lr_start = -5, lr_end = -2):
        
        # initialize variables
        self.train_loader,self.test_loader, self.val_loader = train_loader,test_loader, val_loader
        self.model = model
        self.epochs, self.device, self.model_name, self.optimizer = epochs, device, model_name,optimizer
        self.calculate_accuracy = calculate_accuracy
        self.print_progress = print_every_n_batch
        self.problem_type = problem_type
        self.overwrite_message = overwrite_message
        self.print_result, self.print_full = print_result, print_full
        self.lr_rate_tuning = lr_rate_tuning
        self.clip_batch = clip_batch
        self.clip_batch_size = clip_batch_size
        self.lr_start, self.lr_end = lr_start, lr_end
        
        # create folder to hold model stats
        model_stats_folder = f"{self.model_name} stats"
        if not os.path.exists(model_stats_folder):
            os.makedirs(model_stats_folder)
        self.model_folder = model_stats_folder
        
        # initialize or read past losses
        try:
            past_loss_folder = os.path.join(self.model_folder ,f"{self.model_name} losses")
            past_train_loss_all = list(np.load(past_loss_folder+r"/train_loss_all.npy"))
            past_train_loss = list(np.load(past_loss_folder+r"/train_loss.npy"))
            past_validation_loss = list(np.load(past_loss_folder+r"/validation_loss.npy"))
        except:
            self.losses = {
                "train_loss_all": [],
                "train_loss": [],
                "validation_loss": []
            }
            
            self.accuracy = {
                "train_acc_all": [],
                "train_acc":[],
                "validation_acc": []
            }
        else:
            self.losses = {
                "train_loss_all": past_train_loss_all,
                "train_loss": past_train_loss,
                "validation_loss": past_validation_loss
            }
            if self.calculate_accuracy:
                self.accuracy = {
                    "train_acc_all": list(np.load(past_loss_folder+r"/train_acc_all.npy")),
                    "train_acc": list(np.load(past_loss_folder+r"/train_acc.npy")),
                    "validation_acc": list(np.load(past_loss_folder+r"/validation_acc.npy"))
                }
        
        # update and import loss_functions class
        if update_loss_fn:
            if (print_result & print_full): print("update and import the loss_functions module\n")
            update_file = update_utilities_class(file_name="loss_functions.py",current_path=os.getcwd())
            update_file.run()
        
        from loss_functions import loss_functions_class
        loss_function = loss_functions_class()
        try:
            loss_fn = loss_function.get_loss_fn(self.problem_type)
        except:
            print(f"{self.problem_type} Problem Type is not predefined in the loss_functions_class, need to be added manually")
        self.loss_fn = loss_fn
        
        if (print_result & print_full): print("\nAll initialized, ready to go!")
    
    def lr_tuning(self,dataloader,optimizer,start,end,clip_batch, batch_size):
        assert start-end < 0, "start and end should be negative where start less than end"
        if (self.print_result & self.print_full): print("learning rate tuning\n")
        model = copy.deepcopy(self.model)
        model.train()
        num_step = -(start-end) * 40 + 1
        lre = torch.linspace(start,end,num_step)
        lrs = 10**lre
        lossi = []
        i = 0
        for batch_inputs, batch_labels in dataloader:
            if clip_batch:
                batch_inputs, batch_labels = batch_inputs[:min(batch_size,len(batch_inputs))], batch_labels[:min(batch_size,len(batch_labels))]
            # define the learning rate
            for g in optimizer.param_groups:
                g['lr'] = lrs[i]
            i += 1
            # regular forward pass and backpropogation
            batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
            optimizer.zero_grad()

            model_outputs = model(batch_inputs)
            if "Binary" in self.problem_type:
                model_outputs = model_outputs.squeeze()
                loss = self.loss_fn(model_outputs,batch_labels.float())
            elif len(model_outputs.shape) == 2:
                loss = self.loss_fn(model_outputs,batch_labels)
            else:
                loss = self.loss_fn(torch.flatten(model_outputs,end_dim=1),torch.flatten(batch_labels,end_dim=1))
            loss.backward()
            optimizer.step()
            lossi.append(loss.detach().cpu().item())
            if i == num_step:
                if (self.print_result & self.print_full): print("learning rate tuning finished\n")
                del model
                return lossi

    
    def test(self,mode):
        self.model.eval()
        batch_loss = 0
        batch_acc = 0
        if mode == "validation":
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
            
        with torch.inference_mode():
            for batch_inputs, batch_labels in dataloader:
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
                model_outputs = self.model(batch_inputs)
                if "Binary" in self.problem_type:
                    model_outputs = model_outputs.squeeze()
                    loss = self.loss_fn(model_outputs,batch_labels.float())
                elif len(model_outputs.shape) == 2:
                    loss = self.loss_fn(model_outputs,batch_labels)
                else:
                    loss = self.loss_fn(torch.flatten(model_outputs,end_dim=1),torch.flatten(batch_labels,end_dim=1))
                batch_loss += loss
                if self.calculate_accuracy:
                    if "Binary" in self.problem_type:
                        pred_labels = torch.round(torch.sigmoid(model_outputs))
                    else:
                        pred_labels = model_outputs.argmax(dim=1)
                    acc = torch.eq(pred_labels, batch_labels).sum().item()/len(batch_labels)
                    batch_acc += acc
            avg_loss = batch_loss / len(dataloader)
            avg_acc = batch_acc / len(dataloader) * 100
            if mode != "validation":
                message_file_path = os.path.join(self.model_folder, f"{self.model_name} - Training Information.txt")
                f = open(message_file_path,"a")
                f.write("\n\nTesting Information\n"+"-"*80)
                m = f"Average per-Batch Test Loss: {avg_loss:.4f}"
                print(m)
                f.write("\n"+m)
                if self.calculate_accuracy:
                    m = f"Average per-Batch Test Accuracy: {avg_acc:.2f}%"
                    print(m)
                    f.write("\n"+m)
                f.close()
            return avg_loss, avg_acc
                    
        
    def train(self):
        try:
            lowest_val_loss = np.load(os.path.join(self.model_folder,f"{self.model_name}_lowest_val_loss.npy")).item()
        except:
            lowest_val_loss = 1000000000
        
        start = time.time()
        total_time = 0
        # write the output to a file
        message_file_path = os.path.join(self.model_folder, f"{self.model_name} - Training Information.txt")
        if self.overwrite_message:
            f = open(message_file_path,"w")
        else:
            f = open(message_file_path,"a")
        
        if self.overwrite_message:
            m = f"Basic Specs\n----------------------------------------------------"
            if (self.print_result & self.print_full): print(m)
            f.write("\n"+m)
            sample_inputs, _ = next(iter(self.train_loader))
            m = f"Input Size: {sample_inputs.shape}\n"
            if (self.print_result & self.print_full): print(m)
            f.write("\n"+m)
            m = "\nModel Specs: \n"
            if (self.print_result & self.print_full): print(m)
            f.write("\n"+m)
            num_para = sum(p.numel() for p in self.model.parameters())
            if num_para > 1e6:
                num_para /= 1e6
                unit = "million"
            else:
                num_para /= 1e3
                unit = "thousand"
            m = f"Total number of Parameters {round(num_para,2)} {unit}\n"
            if (self.print_result & self.print_full): print(m)
            f.write("\n"+m)
            if (self.print_result & self.print_full): print(self.model)
            print(self.model,file=f)
            m = "\n\n"
            if (self.print_result & self.print_full): print(m)
            f.write("\n"+m)
        
        f.write("\n\nTraining Information\n" + "-"*80)

        
        # initializing
        num_steps = self.epochs * len(self.train_loader)
        progress_bar = tqdm(range(num_steps))
        print_progress_cycle = 0 # this keeps track of the current number of print_progress cycle
        total_print_progress_cycle = math.ceil(num_steps/self.print_progress)
        

        
        # print initial message
        m = f"Training Begin\n----------------------------------------------------"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        m = f"There are {self.epochs} epochs, and for each epoch, there are {len(self.train_loader)} batches of training data"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        m = f"Total Training Steps: {num_steps}"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        m = f"Total Displaying Information: {total_print_progress_cycle}"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        m = f"Optimizer name - {self.optimizer.__class__.__name__} learning rate: {self.optimizer.param_groups[-1]['lr']}"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        m = f"lowest_val_loss started with {lowest_val_loss}\n"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        
        # initializing
        step = 0
        batch_loss = 0
        batch_acc = 0
        
        # create directory for the model weights
        folder_name = self.model_name + " weights"
        folder_path = os.path.join(self.model_folder,folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if self.lr_rate_tuning:
            lossi = self.lr_tuning(self.train_loader,self.optimizer,self.lr_start,self.lr_end,self.clip_batch,self.clip_batch_size)
            return lossi

        # training loop
        for e in range(self.epochs):
            for batch_inputs, batch_labels in self.train_loader:
                step += 1
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                
                # forward pass
                model_outputs = self.model(batch_inputs)
                
                # loss calculation, backpropogation and update model parameters
                if "Binary" in self.problem_type:
                    model_outputs = model_outputs.squeeze()
                    loss = self.loss_fn(model_outputs,batch_labels.float())
                elif len(model_outputs.shape) == 2:
                    loss = self.loss_fn(model_outputs,batch_labels)
                else:
                    loss = self.loss_fn(torch.flatten(model_outputs,end_dim=1),torch.flatten(batch_labels,end_dim=1))
                loss.backward()
                self.optimizer.step()
                
                # append loss
                self.losses["train_loss_all"].append(loss.detach().cpu().item())
                batch_loss += loss
                
                # if require accuracy calculation
                if self.calculate_accuracy:
                    if "Binary" in self.problem_type:
                        pred_labels = torch.round(torch.sigmoid(model_outputs))
                    else:
                        pred_labels = model_outputs.argmax(dim=1)
                    acc = torch.eq(pred_labels,batch_labels).sum().item()/len(batch_labels)
                    self.accuracy["train_acc_all"].append(acc)
                    batch_acc += acc
                
                # validate and print progress if reach the print_progress or at the last step
                if (step % self.print_progress == 0) | (step == num_steps):
                    print_progress_cycle += 1
                    if (step != num_steps) | (num_steps % self.print_progress == 0):
                        batch_count = self.print_progress
                    else:
                        batch_count = num_steps % self.print_progress
                    
                    # print message
                    m = f"\n\nMessage: {print_progress_cycle} - "\
                          +f"Progress Summary - {batch_count} batches\n--------------------------------"
                    if (self.print_result & self.print_full): print(m)
                    f.write("\n"+m)
                    m = f"Epoch: {e+1} / {self.epochs} || Batch: {step} / {num_steps} || " \
                          + f"Print Cycle: {print_progress_cycle} / {total_print_progress_cycle}"
                    if (self.print_result & self.print_full): print(m)
                    f.write("\n"+m)
                    
                    validation_loss, validation_acc = self.test(mode="validation")
                    self.losses["validation_loss"].append(validation_loss.detach().cpu().item())
                    avg_batch_loss = batch_loss/batch_count
                    self.losses["train_loss"].append(avg_batch_loss.detach().cpu().item())
                    
                    # print message
                    m = f"Average per-Batch Training Loss: {avg_batch_loss:.4f} || " \
                          + f"Average per-Batch Validation Loss: {validation_loss:.4f}"
                    if (self.print_result & (not self.print_full)): 
                        print(f"Batch: {step} / {num_steps} || " + m)
                    if (self.print_result & self.print_full): print(m)
                    f.write("\n"+m)
                    
                    batch_loss = 0
                    
                    # if accuracy need be to calculated
                    if self.calculate_accuracy:
                        self.accuracy["validation_acc"].append(validation_acc)
                        avg_batch_acc = batch_acc/batch_count * 100
                        self.accuracy["train_acc"].append(avg_batch_acc)
                        
                        # print message
                        m = f"Average per-Batch Training Accuracy: {avg_batch_acc:.2f}% || " \
                              + f"Average per-Batch Validation Accuracy: {validation_acc:.2f}%"
                        print(m)
                        if (self.print_result & (not self.print_full)): 
                            print()
                        f.write("\n"+m)
                        
                        batch_acc = 0
                    
                    # calculate model improvement
                    if len(self.losses["train_loss"]) > 1:
                        idx = len(self.losses["train_loss"]) - 1
                        train_loss_perc_decrease = -(self.losses["train_loss"][idx]-self.losses["train_loss"][idx-1]) \
                                                    / self.losses["train_loss"][idx-1] * 100
                        val_loss_perc_decrease = -(self.losses["validation_loss"][idx]-self.losses["validation_loss"][idx-1]) \
                                                 / self.losses["validation_loss"][idx-1] * 100
                        
                        # print message
                        m = "\nModel Improvement\n--------------------------------"
                        if (self.print_result & self.print_full): print(m)
                        f.write("\n"+m)
                        m = f"Average per-Batch Training Loss has decreased by {train_loss_perc_decrease:.2f}%"
                        if (self.print_result & self.print_full): print(m)
                        f.write("\n"+m)
                        m = f"Average per-Batch Validation Loss has decreased by {val_loss_perc_decrease:.2f}%\n"
                        if (self.print_result & self.print_full): print(m)
                        f.write("\n"+m)
                        
                        # if validation loss is the lowest, save the model as the best model weights
                        if validation_loss.cpu() < lowest_val_loss:
                            save_path = folder_path + r"/"+self.model_name+"_best.pth"
                            m = f"Val Loss decreased from {lowest_val_loss:4f} to {validation_loss.cpu():4f} - Saving the Best Model\n"
                            if (self.print_result & self.print_full): print(m)
                            f.write("\n"+m+"\n")
                            torch.save(self.model.state_dict(),save_path)
                            lowest_val_loss = validation_loss.cpu()
                            np.save(os.path.join(self.model_folder,f"{self.model_name}_lowest_val_loss.npy"),lowest_val_loss)
                    end = time.time()
                    time_spent = np.round((end-start)/60,2)
                    total_time += time_spent
                    unit = "minutes"
                    if time_spent > 60:
                        time_spent = np.round(time_spent/60,2)
                        unit = "hours"
                    m = f"This printing cycle took {time_spent} {unit}\n"
                    if (self.print_result & self.print_full): print(m)
                    f.write("\n"+m)
                    start = time.time()
                    
                
                # outside validation and printing
                
                # update progress bar
                progress_bar.update(1)
            
            # outside dataloader iteration
            
        # outside epoch for loop
        save_path = folder_path + r"/"+self.model_name+"_last.pth"
        m = "Saving the Last Model\n"
        if (self.print_result & self.print_full): print(m)
        f.write("\n"+m)
        torch.save(self.model.state_dict(),save_path)
        
        
        # save losses/accuracies
        loss_folder_name = self.model_name + " losses"
        loss_folder_path = os.path.join(self.model_folder,loss_folder_name)
        if not os.path.exists(loss_folder_path):
            os.makedirs(loss_folder_path)
        for key in self.losses:
            np.save(os.path.join(loss_folder_path,key+".npy"),self.losses[key])
        if self.calculate_accuracy:
            for key in self.accuracy:
                np.save(os.path.join(loss_folder_path,key+".npy"),self.accuracy[key])
        
        print("\n All Done\n")
        time_spent = np.round(total_time/60,2)
        m = f"Overall training took {time_spent} hours\n"
        print(m)
        f.write("\n\n"+m+"-"*80+"\n\n\n\n")
        f.close()

    # outside train function

