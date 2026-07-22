import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class Trainer(nn.Module):
    def __init__(self, model, learning_rate, scheduler_coef):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.scheduler_coef = scheduler_coef
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.all_means = []
        self.all_stds = []
        self.all_targets = []
        self.patience = 20         # or pass as argument
        self.early_stop_counter = 0
        self.all_means = []
        self.all_epistemic = []
        self.all_aleatoric = []
        

    def weighted_mse(self, pred, target, weights):
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * weights
        return loss.mean()   

    def run_bayes(self, epochs, training_set, validation_set, kl_weight, MC_samples, **kwargs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=self.scheduler_coef
        )

        for epoch in range(epochs):
            # ---- TRAIN ----
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in training_set:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                # ---- Forward pass ----
                features = model.forward_features(xb)
                mu, sigma, log_var = model.forward_head(features)
                # ---- NLL loss (aleatoric) ----
                nll =  ((yb - mu)**2 / (2 * sigma**2)  + log_var/2).mean()
                
#                 print("mu mean:", model.fcb.weight_mu.mean().item())
#                 print("mu std:", model.fcb.weight_mu.std().item())
#                 print("sigma mean:", torch.log1p(torch.exp(model.fcb.weight_rho)).mean().item())
#                 print("shape : ", model.fcb.weight_mu.shape)
#                 print("===========")

           
                # ---- KL term (Bayesian layer) ----
                kl_loss = self.model.kl_divergence()
#                 print("nll :", nll.item())
#                 print("kl_loss :", kl_loss.item())
                loss = nll + kl_weight * kl_loss
                
                # ---- Backprop ----
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # ---- Logging ----
            epoch_loss /= len(training_set)
            self.train_losses.append(epoch_loss)
            
            
            # ---- VALIDATION ----
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in validation_set:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    # ---- Forward deterministic part ONCE ----
                    features = model.forward_features(xb)

                    # ---- Monte Carlo sampling (Bayesian head only) ----
                    mus = []
                    sigmas = []
                    log_vars = []

                    for _ in range(MC_samples):
                        mu, sigma, log_var = model.forward_head(features)
                        mus.append(mu)
                        sigmas.append(sigma)
                        log_vars.append(log_var)

                    mus = torch.stack(mus)         # [N, batch, dim]
                    sigmas = torch.stack(sigmas)   # [N, batch, dim]
                    log_vars = torch.stack(log_vars)# [N, batch, dim]

                    # ---- Mean prediction ----
                    mean = mus.mean(dim=0)

                    # ---- Uncertainty decomposition ----
                    epistemic_var = mus.var(dim=0)
                    aleatoric_var = (sigmas ** 2).mean(dim=0)

                    total_var = epistemic_var + aleatoric_var
                    total_std = torch.sqrt(total_var)
                    
                    self.all_means.append(mean.cpu())
                    self.all_epistemic.append(epistemic_var.cpu())
                    self.all_aleatoric.append(aleatoric_var.cpu())

                    # ---- Proper NLL loss (Monte Carlo estimate) ----
                    # average NLL over MC samples
                    nll =  ((mus - yb)**2 / ( 2 * sigmas**2) + log_vars/2).mean()

                    val_loss += nll.item()

            # ---- Normalize loss ----
            val_loss /= len(validation_set)
            self.val_losses.append(val_loss)

            scheduler.step()

            # ---- Save best model ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = self.model.state_dict()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}: train={self.train_losses[-1]:.4f}, val={self.val_losses[-1]:.4f}")

            # ---- EARLY STOPPING ----
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        all_means = torch.cat(self.all_means, dim=0)
        all_epistemic = torch.cat(self.all_epistemic, dim=0)
        all_aleatoric = torch.cat(self.all_aleatoric, dim=0)
        
        # ---- Save model ----
        torch.save({
            "model": self.best_weights,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,

            # ---- predictions + uncertainties ----
            "val_mu": all_means,
            "val_epistemic": all_epistemic,
            "val_aleatoric": all_aleatoric,
        }, f"{model.model_name}_3.pth")
        
        
####################################################### Training for U-Net

    def run_unet(self, epochs, training_set, validation_set, **kwargs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate) #, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=4, gamma=self.scheduler_coef
        )

        for epoch in range(epochs):
            # ---- TRAIN ----
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in training_set:
                
                xb, xout = xb[0].to(device), xb[1].to(device)
                yb, _ = yb[0].to(device), yb[1]
                
                optimizer.zero_grad()

                # ---- Forward pass ----
                outputs = model(xb)
                loss = F.mse_loss(outputs, xout)
                
                # ---- Backprop ----
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # ---- Logging ----
            epoch_loss /= len(training_set)
            self.train_losses.append(epoch_loss)
            
            
            # ---- VALIDATION ----
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in validation_set:

                    xb, xout = xb[0].to(device), xb[1].to(device)
                    yb, _ = yb[0].to(device), yb[1]

                    outputs = model(xb)
                    loss = F.mse_loss(outputs, xout)
                    val_loss += loss.item()

                self.val_losses.append(val_loss / len(validation_set))

            scheduler.step()

            # ---- Save best model ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = copy.deepcopy(model.state_dict())
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}: train={self.train_losses[-1]:.4f}, val={self.val_losses[-1]:.4f}")

            # ---- EARLY STOPPING ----
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
                
        # ---- Save model ----
        torch.save({
            "model": self.best_weights,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, f"{model.model_name}_4.pth")
        
####################################################################### Training for U Net then Resnet

    def run_unet_resnet(self, epochs, training_set, validation_set, **kwargs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

#         optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)#, weight_decay=1e-2)
#         scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=4, gamma=self.scheduler_coef
#         )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100)

        for epoch in range(epochs):
            # ---- TRAIN ----
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in training_set:
                
                xb, xout = xb[0].to(device), xb[1].to(device)
                xb = xb.unsqueeze(1)
                xout = xout.unsqueeze(1)
                yb,_ = yb[0].to(device), yb[1]
                # yb = yb.unsqueeze(1)
                optimizer.zero_grad()

#                 # ---- Forward pass ----
                out_lens = model.forward_unet(xb)
# #                 out_lens = torch.cat([out_lens, xb - out_lens], dim=1)
                outputs = model.forward_resnet(out_lens)
                # outputs = model(xb)
                
                loss1 = F.mse_loss(out_lens, xout)
                loss2 = F.mse_loss(outputs, yb)
                loss =  loss1 + 0.1 * loss2
                
                
                # ---- Backprop ----
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # ---- Logging ----
            epoch_loss /= len(training_set)
            self.train_losses.append(epoch_loss)
            
            
            # ---- VALIDATION ----
            model.eval()
            val_loss = 0.0
            loss_1 = 0
            loss_2 = 0

            with torch.no_grad():
                for xb, yb in validation_set:

                    xb, xout = xb[0].to(device), xb[1].to(device)
                    xb = xb.unsqueeze(1)
                    xout = xout.unsqueeze(1)
                    yb, _ = yb[0].to(device), yb[1]
                    
                    # ---- Forward pass ----
                    out_lens = model.forward_unet(xb)
#                     out_lens = torch.cat([out_lens, xb - out_lens], dim=1)
                    outputs = model.forward_resnet(out_lens)
                    # outputs = model(xb)

                    loss1 = F.mse_loss(out_lens, xout)
                    loss2 = F.mse_loss(outputs, yb)
                    loss = loss1 + 0.1 * loss2
                    # loss_1 += loss1.item()
                    # loss_2 += loss2.item()
                    val_loss += loss.item()

                self.val_losses.append(val_loss / len(validation_set))

            scheduler.step()

            # ---- Save best model ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = copy.deepcopy(model.state_dict())
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}: train={self.train_losses[-1]:.4f}, val={self.val_losses[-1]:.4f}")
            # print(f"Validation loss 1 {loss_1 / len(validation_set)}, Validation loss 2 {loss_2/ len(validation_set)}")

            # ---- EARLY STOPPING ----
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
                
        # ---- Save model ----
        torch.save({
            "model": self.best_weights,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, f"{model.model_name}_euclid_12.pth")

        
        
        

###############################################################U Net and Classical NN

    def run_unet_NN(self, epochs, training_set, validation_set, **kwargs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50)

#         optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
#         scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=10, gamma=self.scheduler_coef
#         )

        for epoch in range(epochs):
            # ---- TRAIN ----
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in training_set:
                
                xb, xout = xb[0].to(device), xb[1].to(device)
                yb = yb.to(device)
                
                optimizer.zero_grad()

                # ---- Forward pass ----
                out_lens, x_latent = model.forward_unet(xb, return_latent=True)
                outputs = model.forward_latentNN(x_latent)
                
                loss1 = F.mse_loss(out_lens, xout)
                loss2 = F.mse_loss(outputs, yb)
                loss = loss1 + loss2
                
                # ---- Backprop ----
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # ---- Logging ----
            epoch_loss /= len(training_set)
            self.train_losses.append(epoch_loss)
            
            
            # ---- VALIDATION ----
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in validation_set:

                    xb, xout = xb[0].to(device), xb[1].to(device)
                    yb = yb.to(device)

                    out_lens, x_latent = model.forward_unet(xb, return_latent=True)
                    outputs = model.forward_latentNN(x_latent)

                    loss1 = F.mse_loss(out_lens, xout)
                    loss2 = F.mse_loss(outputs, yb)
                    loss = loss1 + loss2
                    
                    val_loss += loss.item()

                self.val_losses.append(val_loss / len(validation_set))

            scheduler.step()

            # ---- Save best model ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = copy.deepcopy(model.state_dict())
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}: train={self.train_losses[-1]:.4f}, val={self.val_losses[-1]:.4f}")

            # ---- EARLY STOPPING ----
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # ---- Save model ----
        torch.save({
            "model": self.best_weights,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, f"{model.model_name}_4.pth")
