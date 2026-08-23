import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict
import io
import contextlib
from agents.base_agent import BaseAgent



class BrinkmanNormalization:
    """
    Normalization class for Brinkman momentum and heat transfer equations.
    """
    def __init__(self, params):
        self.mu, self.epsilon_p, self.kappa, self.rho_f = params['mu'], params['epsilon_p'], params['kappa'], params['rho_f']
        self.beta, self.T_ref, self.C_p_f, self.k_eff = params['beta'], params['T_ref'], params['C_p_f'], params['k_eff']
        self.g, self.T_h, self.T_c = params['g'], params['T_h'], params['T_c']

        self.L_x, self.L_y = 0.1, 0.1
        self.T_max, self.T_min = 353.15, 293.15
        self.U_max, self.U_min = 0.00032752, -0.00033066
        self.V_max, self.V_min = 0.00070135, -0.00067529
        self.P_max, self.P_min = 101330, 100340
        self.T_scale = (self.T_max - self.T_min) / 2
        self.T_mean = (self.T_max + self.T_min) / 2
        self.U_scale = max(abs(self.U_max), abs(self.U_min))
        self.V_scale = max(abs(self.V_max), abs(self.V_min))
        self.P_scale = (self.P_max - self.P_min) / 2
        self.P_mean = (self.P_max + self.P_min) / 2
        self.L_scale = self.L_x

    def normalize_coordinates(self, x, y):
        return 2 * x / self.L_x - 1, 2 * y / self.L_y - 1

    def denormalize_coordinates(self, x_norm, y_norm):
        return (x_norm + 1) * self.L_x / 2, (y_norm + 1) * self.L_y / 2

    def denormalize_variables(self, T_norm, P_norm, u_x_norm, u_y_norm):
        T = T_norm * self.T_scale + self.T_mean
        P = P_norm * self.P_scale + self.P_mean
        u_x = u_x_norm * self.U_scale
        u_y = u_y_norm * self.V_scale
        return T, P, u_x, u_y

    def normalized_momentum_x_loss(self, derivs):
        """
        Computes the residual of the full x-momentum equation (Brinkman with convection).
        The key change is the addition of the non-linear convective term.
        Residual = Convection + PressureGradient + Darcy - Viscous = 0
        """
        dx_scale, dy_scale = 2 / self.L_x, 2 / self.L_y
        u_x = derivs['u_x_norm'] * self.U_scale
        u_y = derivs['u_y_norm'] * self.V_scale
        du_x_dx = derivs['du_x_dx_norm'] * self.U_scale * dx_scale
        du_x_dy = derivs['du_x_dy_norm'] * self.U_scale * dy_scale
        dP_dx = derivs['dP_dx_norm'] * self.P_scale * dx_scale
        d2u_x_dx2 = derivs['d2u_x_dx2_norm'] * self.U_scale * dx_scale**2
        d2u_x_dy2 = derivs['d2u_x_dy2_norm'] * self.U_scale * dy_scale**2
        term_convection = (self.rho_f / self.epsilon_p**2) * (u_x * du_x_dx + u_y * du_x_dy)
        term_pressure = dP_dx
        term_darcy = (self.mu / self.kappa) * u_x
        term_viscous = -(self.mu / self.epsilon_p) * (d2u_x_dx2 + d2u_x_dy2)
        residual = term_convection + term_pressure + term_darcy + term_viscous
        normalization_scale = self.P_scale / self.L_scale
        return residual / normalization_scale


    def normalized_momentum_y_loss(self, derivs):
        """
        Computes the residual of the full y-momentum equation (Brinkman with convection and buoyancy).
        Key changes:
        1. Addition of the non-linear convective term.
        2. Using the standard Boussinesq approximation for the buoyancy term.
        Residual = Convection + PressureGradient + Darcy - Viscous - Buoyancy = 0
        """
        dx_scale, dy_scale = 2 / self.L_x, 2 / self.L_y
        u_x = derivs['u_x_norm'] * self.U_scale
        u_y = derivs['u_y_norm'] * self.V_scale
        T = derivs['T_norm'] * self.T_scale + self.T_mean
        du_y_dx = derivs['du_y_dx_norm'] * self.V_scale * dx_scale
        du_y_dy = derivs['du_y_dy_norm'] * self.V_scale * dy_scale
        dP_dy = derivs['dP_dy_norm'] * self.P_scale * dy_scale
        d2u_y_dx2 = derivs['d2u_y_dx2_norm'] * self.V_scale * dx_scale**2
        d2u_y_dy2 = derivs['d2u_y_dy2_norm'] * self.V_scale * dy_scale**2
        term_convection = -(self.rho_f / self.epsilon_p**2) * (u_x * du_y_dx + u_y * du_y_dy)
        term_pressure = -dP_dy
        term_darcy = -(self.mu / self.kappa) * u_y
        term_viscous = (self.mu / self.epsilon_p) * (d2u_y_dx2 + d2u_y_dy2)
        term_buoyancy = -self.rho_f * self.g * self.beta * (T - self.T_ref)
        residual = term_convection + term_pressure + term_darcy + term_viscous + term_buoyancy
        normalization_scale = self.P_scale / self.L_scale
        return residual / normalization_scale

    def normalized_heat_loss(self, derivs):
        dx_scale, dy_scale = 2 / self.L_x, 2 / self.L_y
        dT_dx = derivs['dT_dx_norm'] * self.T_scale * dx_scale
        dT_dy = derivs['dT_dy_norm'] * self.T_scale * dy_scale
        d2T_dx2 = derivs['d2T_dx2_norm'] * self.T_scale * dx_scale**2
        d2T_dy2 = derivs['d2T_dy2_norm'] * self.T_scale * dy_scale**2
        u_x = derivs['u_x_norm'] * self.U_scale
        u_y = derivs['u_y_norm'] * self.V_scale
        term_advection = self.rho_f * self.C_p_f * (u_x * dT_dx + u_y * dT_dy)
        term_diffusion = self.k_eff * (d2T_dx2 + d2T_dy2)
        residual = term_advection - term_diffusion
        characteristic_scale = self.k_eff * self.T_scale / (self.L_scale**2)
        return residual / characteristic_scale

    def normalized_continuity_loss(self, derivs):
        dx_scale, dy_scale = 2 / self.L_x, 2 / self.L_y
        du_x_dx = derivs['du_x_dx_norm'] * self.U_scale * dx_scale
        du_y_dy = derivs['du_y_dy_norm'] * self.V_scale * dy_scale
        divergence = du_x_dx + du_y_dy
        char_vel_grad = max(self.U_scale / self.L_x, self.V_scale / self.L_y) if self.L_x > 0 else 1.0
        return divergence / char_vel_grad

    def get_normalized_bcs(self):
        return {
            'T_left': (self.T_h - self.T_mean) / self.T_scale,
            'T_right': (self.T_c - self.T_mean) / self.T_scale,
            'P_ref': (self.P_max - self.P_mean) / self.P_scale
        }
    

class PINNModel(nn.Module):
    def __init__(self, normalizer, layers):
        super().__init__()
        self.normalizer = normalizer
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x_norm, y_norm):
        H = torch.cat([x_norm, y_norm], dim=1)
        for layer in self.layers[:-1]:
            H = torch.tanh(layer(H))
        output = torch.tanh(self.layers[-1](H))
        return output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]

    def compute_derivatives(self, x_norm, y_norm):
        x_norm.requires_grad_(True)
        y_norm.requires_grad_(True)
        T_norm, P_norm, u_x_norm, u_y_norm = self.forward(x_norm, y_norm)
        grads = torch.autograd.grad
        T_sum, P_sum, ux_sum, uy_sum = T_norm.sum(), P_norm.sum(), u_x_norm.sum(), u_y_norm.sum()

        dT_dx_n, dT_dy_n = grads(T_sum, (x_norm, y_norm), create_graph=True)
        dP_dx_n, dP_dy_n = grads(P_sum, (x_norm, y_norm), create_graph=True)
        dux_dx_n, dux_dy_n = grads(ux_sum, (x_norm, y_norm), create_graph=True)
        duy_dx_n, duy_dy_n = grads(uy_sum, (x_norm, y_norm), create_graph=True)

        d2T_dx2_n = grads(dT_dx_n.sum(), x_norm, create_graph=True)[0]
        d2T_dy2_n = grads(dT_dy_n.sum(), y_norm, create_graph=True)[0]
        d2ux_dx2_n = grads(dux_dx_n.sum(), x_norm, create_graph=True)[0]
        d2ux_dy2_n = grads(dux_dy_n.sum(), y_norm, create_graph=True)[0]
        d2uy_dx2_n = grads(duy_dx_n.sum(), x_norm, create_graph=True)[0]
        d2uy_dy2_n = grads(duy_dy_n.sum(), y_norm, create_graph=True)[0]
        d2ux_dxdy_n = grads(dux_dy_n.sum(), x_norm, create_graph=True)[0]
        d2uy_dxdy_n = grads(duy_dx_n.sum(), y_norm, create_graph=True)[0]

        return {
            'T_norm': T_norm, 'P_norm': P_norm, 'u_x_norm': u_x_norm, 'u_y_norm': u_y_norm,
            'dT_dx_norm': dT_dx_n, 'dT_dy_norm': dT_dy_n, 'dP_dx_norm': dP_dx_n, 'dP_dy_norm': dP_dy_n,
            'du_x_dx_norm': dux_dx_n, 'du_x_dy_norm': dux_dy_n, 'du_y_dx_norm': duy_dx_n, 'du_y_dy_norm': duy_dy_n,
            'd2T_dx2_norm': d2T_dx2_n, 'd2T_dy2_norm': d2T_dy2_n, 'd2u_x_dx2_norm': d2ux_dx2_n, 'd2u_x_dy2_norm': d2ux_dy2_n,
            'd2u_y_dx2_norm': d2uy_dx2_n, 'd2u_y_dy2_norm': d2uy_dy2_n, 'd2u_x_dxdy_norm': d2ux_dxdy_n, 'd2u_y_dxdy_norm': d2uy_dxdy_n
        }
    

class PINNTrainer:
    def __init__(self, model, normalizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.normalizer = normalizer
        self.device = device
        self.loss_weights = {}
        self.history = defaultdict(list)

    def generate_collocation_points(self, n_interior, n_boundary):
        """
        Generate training points using Sobol sequence for the interior
        and linspace for boundaries.
        """
        # 1. Create a Sobol engine for 2 dimensions (x, y)
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)

        # 2. Generate n_interior points in the [0, 1]^2 range
        interior_points_unit = sobol_engine.draw(n_interior)

        # 3. Map points from [0, 1]^2 to [-1, 1]^2 for the normalized domain
        interior_points = interior_points_unit * 2 - 1

        # --- Boundary points (linspace is still best for boundaries) ---
        n_per_edge = n_boundary // 4
        y_edge = torch.linspace(-1, 1, n_per_edge).view(-1, 1)
        left_points = torch.cat([torch.full_like(y_edge, -1), y_edge], dim=1)
        right_points = torch.cat([torch.full_like(y_edge, 1), y_edge], dim=1)

        x_edge = torch.linspace(-1, 1, n_per_edge).view(-1, 1)
        bottom_points = torch.cat([x_edge, torch.full_like(x_edge, -1)], dim=1)
        top_points = torch.cat([x_edge, torch.full_like(x_edge, 1)], dim=1)

        return {
            'interior': interior_points.to(self.device),
            'left': left_points.to(self.device),
            'right': right_points.to(self.device),
            'bottom': bottom_points.to(self.device),
            'top': top_points.to(self.device)
        }

    def compute_pde_losses(self, points):
        x_norm, y_norm = points[:, 0:1], points[:, 1:2]
        derivatives = self.model.compute_derivatives(x_norm, y_norm)
        return {
            'momentum_x': self.normalizer.normalized_momentum_x_loss(derivatives).pow(2).mean(),
            'momentum_y': self.normalizer.normalized_momentum_y_loss(derivatives).pow(2).mean(),
            'heat': self.normalizer.normalized_heat_loss(derivatives).pow(2).mean(),
            'continuity': self.normalizer.normalized_continuity_loss(derivatives).pow(2).mean()
        }

    def compute_bc_losses(self, boundary_points):
        """
        Computes the boundary condition losses for the natural convection problem.
        The key change is using a single pressure reference point instead of a pressure boundary.
        """
        losses = {}
        bcs = self.normalizer.get_normalized_bcs()

        # --- Standard Boundaries ---

        # Left boundary (x=0): T=T_h, u=v=0
        x_l, y_l = boundary_points['left'][:, 0:1], boundary_points['left'][:, 1:2]
        T_l_pred, _, u_l_pred, v_l_pred = self.model(x_l, y_l)
        losses['bc_T_left'] = (T_l_pred - bcs['T_left']).pow(2).mean()
        losses['bc_u_left'] = u_l_pred.pow(2).mean() + v_l_pred.pow(2).mean()

        # Right boundary (x=0.1): T=T_c, u=v=0
        x_r, y_r = boundary_points['right'][:, 0:1], boundary_points['right'][:, 1:2]
        T_r_pred, _, u_r_pred, v_r_pred = self.model(x_r, y_r)
        losses['bc_T_right'] = (T_r_pred - bcs['T_right']).pow(2).mean()
        losses['bc_u_right'] = u_r_pred.pow(2).mean() + v_r_pred.pow(2).mean()

        # Bottom boundary (y=0): dT/dy=0 (insulated), u=v=0
        x_b, y_b = boundary_points['bottom'][:, 0:1].requires_grad_(True), boundary_points['bottom'][:, 1:2].requires_grad_(True)
        T_b_pred, _, u_b_pred, v_b_pred = self.model(x_b, y_b)
        dT_dy_b = torch.autograd.grad(T_b_pred.sum(), y_b, create_graph=True)[0]
        losses['bc_dT_dy_bottom'] = dT_dy_b.pow(2).mean()
        losses['bc_u_bottom'] = u_b_pred.pow(2).mean() + v_b_pred.pow(2).mean()
        # REMOVED: The pressure condition on the entire bottom boundary is no longer needed.
        # losses['bc_p_bottom'] = (P_b - bcs['P_bottom']).pow(2).mean()

        # Top boundary (y=0.1): dT/dy=0 (insulated), u=v=0
        x_t, y_t = boundary_points['top'][:, 0:1].requires_grad_(True), boundary_points['top'][:, 1:2].requires_grad_(True)
        T_t_pred, _, u_t_pred, v_t_pred = self.model(x_t, y_t)
        dT_dy_t = torch.autograd.grad(T_t_pred.sum(), y_t, create_graph=True)[0]
        losses['bc_dT_dy_top'] = dT_dy_t.pow(2).mean()
        losses['bc_u_top'] = u_t_pred.pow(2).mean() + v_t_pred.pow(2).mean()

        # Get device and dtype from an existing tensor to ensure consistency
        device = x_l.device
        dtype = x_l.dtype

        ref_point_coords = torch.tensor([[-1.0, -1.0]], device=device, dtype=dtype)

        _, P_ref_pred, _, _ = self.model(ref_point_coords[:, 0:1], ref_point_coords[:, 1:2])

        # Add the loss for the pressure reference point
        losses['bc_p_ref'] = (P_ref_pred - bcs['P_ref']).pow(2).mean()

        return losses

    def compute_total_loss(self, collocation_points):
        pde_losses = self.compute_pde_losses(collocation_points['interior'])
        bc_losses = self.compute_bc_losses(collocation_points) # زیان‌های مرزی اینجا تعریف شده

        total_loss = 0.0
        loss_components = {}
        all_losses = {**pde_losses, **bc_losses}

        loss_components['bc_T'] = bc_losses.get('bc_T_left', 0) + bc_losses.get('bc_T_right', 0)
        loss_components['bc_u'] = bc_losses.get('bc_u_left', 0) + bc_losses.get('bc_u_right', 0) + bc_losses.get('bc_u_top', 0) + bc_losses.get('bc_u_bottom', 0)
        loss_components['bc_p'] = bc_losses.get('bc_p_ref', 0)
        loss_components['bc_neumann'] = bc_losses.get('bc_dT_dy_top', 0) + bc_losses.get('bc_dT_dy_bottom', 0)

        loss_components.update(pde_losses)

        for key, value in loss_components.items():
            weight = self.loss_weights.get(key, 1.0)

            total_loss += weight * value

        return total_loss, {k: v.item() for k, v in loss_components.items()}

    def train(self, epochs_adam, epochs_lbfgs, n_interior,
              n_boundary, lr, print_every, plot_every):
        collocation_points = self.generate_collocation_points(n_interior, n_boundary)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=500, min_lr=1e-7)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


        print(f"Starting training on {self.device}...")
        start_time = time.time()
        best_loss = float('inf')

        progress_bar = tqdm(range(epochs_adam), desc="Training")
        for epoch in progress_bar:
            self.model.train()
            loss, loss_components = self.compute_total_loss(collocation_points)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)

            weighted_losses = {k: v * self.loss_weights.get(k, 1.0) for k, v in loss_components.items()}
            self.history['loss'].append(loss.item())
            # for key, value in loss_components.items():
            #     self.history[key].append(value)
            for key, value in weighted_losses.items():
                self.history[key].append(value)

            progress_bar.set_postfix({'loss': f'{loss.item():.4e}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                            'loss': best_loss}, './results/best_model.pth')

            if epoch == 0:
                initial_losses = ''
                for key, value in sorted(weighted_losses.items()):
                    initial_losses += f"  - {key:20s}: {value:.6e}\n"
            elif (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                # Clear progress bar line before printing details
                print("\r" + " " * (progress_bar.ncols or 80))
                print(f"\n--- Epoch {epoch+1}/{epochs_adam} | Time: {elapsed:.1f}s ---")
                print(f"Total Loss: {loss.item():.6e} | Best Loss: {best_loss:.6e}")
                print("Loss Components:")
                # for key, value in sorted(loss_components.items()):
                #     print(f"  - {key:20s}: {value:.6e}")
                # print('%%%%%%%%%')
                for key, value in sorted(weighted_losses.items()):
                    print(f"  - {key:20s}: {value:.6e}")
                print("-" * 40)

            if (epoch + 1) % plot_every == 0 and epoch > 0:
                self.plot_training_progress()
                self.plot_solution_fields(epoch+1)

            # Adaptive resampling (optional)
            if (epoch + 1) % 2000 == 0 and epoch > 0:
                print("\nResampling collocation points...")
                collocation_points = self.generate_collocation_points(n_interior, n_boundary)

        print(f"Training completed in {time.time() - start_time:.1f}s. Best loss: {best_loss:.4e}")
        print(f"\nAdam training finished after {epochs_adam} epochs. Best loss: {best_loss:.4e}")

        print("\n--- Loading best model and starting L-BFGS Optimization ---")
                
        checkpoint = torch.load('./results/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=1.0, 
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _ = self.compute_total_loss(collocation_points)
            loss.backward()
            return loss

        progress_bar_lbfgs = tqdm(range(epochs_lbfgs), desc="L-BFGS Training")
        for epoch in progress_bar_lbfgs:
            self.model.train()

            loss = optimizer_lbfgs.step(closure)

            self.history['loss'].append(loss.item())
            _, loss_components = self.compute_total_loss(collocation_points) 
            weighted_losses = {k: v * self.loss_weights.get(k, 1.0) for k, v in loss_components.items()}
            for key, value in weighted_losses.items():
                self.history[key].append(value)

            progress_bar_lbfgs.set_postfix({'loss': f'{loss.item():.4e}'})

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({'epoch': epoch + epochs_adam, 'model_state_dict': self.model.state_dict(),
                            'loss': best_loss}, './results/best_model.pth')

            if (epoch + 1) % print_every == 0: 
                print(f"\nL-BFGS Epoch {epoch+1}/{epochs_lbfgs} | Loss: {loss.item():.6e}")
                print("Loss Components:")
                # for key, value in sorted(loss_components.items()):
                #     print(f"  - {key:20s}: {value:.6e}")
                # print('%%%%%%%%%')
                for key, value in sorted(weighted_losses.items()):
                    print(f"  - {key:20s}: {value:.6e}")
                print("-" * 40)
            if (epoch + 1) % plot_every == 0:
                self.plot_training_progress()
                self.plot_solution_fields(epoch + 1 + epochs_adam)
            # Adaptive resampling (optional)
            if (epoch + 1) % 200 == 0 and epoch > 0:
                print("\nResampling collocation points...")
                collocation_points = self.generate_collocation_points(n_interior, n_boundary)

        print(f"\nTraining completed in {time.time() - start_time:.1f}s. Final best loss: {best_loss:.4e}")

        return initial_losses

    def plot_training_progress(self):
        """
        Plot training history (Original version with 2x3 grid)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Total loss
        axes[0, 0].semilogy(self.history['loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

        # PDE losses
        pde_keys = ['momentum_x', 'momentum_y', 'heat', 'continuity']
        for key in pde_keys:
            if key in self.history:
                axes[0, 1].semilogy(self.history[key], label=key)
        axes[0, 1].set_title('PDE Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # BC losses
        bc_keys = [k for k in self.history.keys() if 'bc_' in k]
        # Plot a subset of BC losses to avoid clutter
        for key in bc_keys[:6]:
            axes[0, 2].semilogy(self.history[key], label=key.replace('bc_', ''))
        axes[0, 2].set_title('BC Losses (subset)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].grid(True)

        # Loss components pie chart (latest values)
        latest_components = {k: self.history[k][-1] for k in self.history.keys() if k != 'loss'}
        # Filter out very small values for better visualization
        filtered_components = {k: v for k, v in latest_components.items() if v > 1e-9}
        keys = list(filtered_components.keys())
        values = list(filtered_components.values())
        if values:
            axes[1, 0].pie(values, labels=keys, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Loss Component Distribution (Latest)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No significant loss components', ha='center', va='center')


        # Learning rate or another plot
        # Using "Loss every 10 epochs" plot from original code
        if len(self.history['loss']) > 10:
            axes[1, 1].plot(self.history['loss'][::10])
            axes[1, 1].set_title('Loss (every 10 epochs)')
            axes[1, 1].set_xlabel('Epoch (x10)')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough epochs for this plot', ha='center', va='center')

        # Remove empty subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()
        plt.savefig('./results/training_progress_detailed.png', dpi=100)
        plt.close()

    def plot_solution_fields(self, epoch, final_epoch=False):
        self.model.eval()
        n_plot = 50
        x, y = torch.linspace(-1, 1, n_plot), torch.linspace(-1, 1, n_plot)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        x_flat, y_flat = X.reshape(-1, 1).to(self.device), Y.reshape(-1, 1).to(self.device)
        with torch.no_grad():
            T_n, P_n, ux_n, uy_n = self.model(x_flat, y_flat)
            T, P, u_x, u_y = self.normalizer.denormalize_variables(
                T_n.cpu(), P_n.cpu(), ux_n.cpu(), uy_n.cpu())
        T, P, u_x, u_y = [v.reshape(n_plot, n_plot).numpy() for v in [T, P, u_x, u_y]]
        u_mag = np.sqrt(u_x**2 + u_y**2)
        X_p, Y_p = self.normalizer.denormalize_coordinates(X.numpy(), Y.numpy())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Solution Fields at Epoch {epoch}')

        def plot_contour(ax, X, Y, Z, title, cmap):
            im = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
            ax.set_title(title); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
            fig.colorbar(im, ax=ax)

        plot_contour(axes[0, 0], X_p, Y_p, T, 'Temperature (K)', 'hot')
        plot_contour(axes[0, 1], X_p, Y_p, P, 'Pressure (Pa)', 'viridis')
        plot_contour(axes[0, 2], X_p, Y_p, u_mag, 'Velocity Magnitude (m/s)', 'plasma')
        plot_contour(axes[1, 0], X_p, Y_p, u_x, 'u_x (m/s)', 'RdBu')
        plot_contour(axes[1, 1], X_p, Y_p, u_y, 'u_y (m/s)', 'RdBu')
        axes[1, 2].streamplot(X_p, Y_p, u_x, u_y, color=u_mag, cmap='plasma', density=1.5)
        axes[1, 2].set_title('Streamlines'); axes[1, 2].set_xlabel('x (m)'); axes[1, 2].set_ylabel('y (m)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if final_epoch:
            plt.savefig(f'./results/pinnPlots.png', dpi=100)
        else:
            plt.savefig(f'./results/solution_epoch_{epoch}.png', dpi=100)
        plt.close()

    def validate(self, test_points=None):
        self.model.eval()
        if test_points is None:
            n_test = 1000
            test_points = (torch.rand(n_test, 2) * 2 - 1).to(self.device)
        pde_losses = self.compute_pde_losses(test_points)
        print("\nValidation PDE Residuals (on random points):")
        for key, value in pde_losses.items():
            print(f"  - {key}: {value.item():.6e}")
        return pde_losses

    def compute_l2_error(self, test_df):
        self.model.eval()
        # try: test_df = pd.read_csv(test_data_path)
        # except FileNotFoundError: print(f"Error: File not found at {test_data_path}"); return {}

        x_phys = torch.tensor(test_df['x'].values, dtype=torch.float32).view(-1, 1).to(self.device)
        y_phys = torch.tensor(test_df['y'].values, dtype=torch.float32).view(-1, 1).to(self.device)
        x_norm, y_norm = self.normalizer.normalize_coordinates(x_phys, y_phys)

        with torch.no_grad():
            T_n, P_n, ux_n, uy_n = self.model(x_norm, y_norm)
            T_p, P_p, ux_p, uy_p = self.normalizer.denormalize_variables(
                T_n.cpu(), P_n.cpu(), ux_n.cpu(), uy_n.cpu())

        T_t = torch.tensor(test_df['T'].values, dtype=torch.float32).view(-1, 1)
        P_t = torch.tensor(test_df['P'].values, dtype=torch.float32).view(-1, 1)
        ux_t = torch.tensor(test_df['U'].values, dtype=torch.float32).view(-1, 1)
        uy_t = torch.tensor(test_df['V'].values, dtype=torch.float32).view(-1, 1)

        # L2 error
        print("\n" + "="*50 + "\nCalculating L2 Relative Error from CSV...\n" + "="*50)
        def l2_rel_err(true, pred): return torch.linalg.norm(true - pred) / torch.linalg.norm(true)
        l2_errors = {'T': l2_rel_err(T_t, T_p).item(), 'P': l2_rel_err(P_t, P_p).item(),
                'u_x': l2_rel_err(ux_t, ux_p).item(), 'u_y': l2_rel_err(uy_t, uy_p).item()}

        print(f"  - Temperature (T): {l2_errors['T']:.4e}\n  - Pressure (P):    {l2_errors['P']:.4e}")
        print(f"  - Velocity (u_x):  {l2_errors['u_x']:.4e}\n  - Velocity (u_y):  {l2_errors['u_y']:.4e}")
        print("="*50)

        mean_relative_error = (l2_errors['T'] + l2_errors['P']) / 2

        # MAPE error
        print("\n" + "="*50 + "\nCalculating MAPE Error from CSV...\n" + "="*50)
        def mape_err(true, pred, eps=1e-18): return torch.mean(torch.abs((true - pred) / (true + eps))) * 100
        mape_errors = {'T': mape_err(T_t, T_p).item(), 'P': mape_err(P_t, P_p).item(),
                'u_x': mape_err(ux_t, ux_p).item(), 'u_y': mape_err(uy_t, uy_p).item()}

        print(f"  - Temperature (T): {mape_errors['T']:.4e}\n  - Pressure (P):    {mape_errors['P']:.4e}")
        print(f"  - Velocity (u_x):  {mape_errors['u_x']:.4e}\n  - Velocity (u_y):  {mape_errors['u_y']:.4e}")
        print("="*50)
        
        return l2_errors, mean_relative_error

    
class TrainerAgent(BaseAgent):
    def assign_loss_weights(self, classification_dict):
        # mapping rules
        equation_weights = {
            "hard": 100000,
            "medium": 10,
            "easy": 1
        }
        bc_weights = {
            "hard": 10000,
            "medium": 1000,
            "easy": 1
        }

        weight_dict = {}

        for key, value in classification_dict.items():
            val_lower = value.lower()

            # detect if it's a boundary condition
            if key.startswith("bc_"):
                if "hard" in val_lower:
                    weight_dict[key] = bc_weights["hard"]
                elif "medium" in val_lower:
                    weight_dict[key] = bc_weights["medium"]
                else:
                    weight_dict[key] = bc_weights["easy"]

            # otherwise it's a governing equation
            else:
                if "hard" in val_lower:
                    weight_dict[key] = equation_weights["hard"]
                elif "medium" in val_lower:
                    weight_dict[key] = equation_weights["medium"]
                else:
                    weight_dict[key] = equation_weights["easy"]

        return weight_dict



    def run(self, state: dict) -> dict:
        print("------ Trainer Agent ------")

        layer_info = state.pinn_architecture["neurons_per_layer"]
        # loss_weights_info = state.pinn_architecture["loss_weighting"]
        # validation_data = state.validation_data

        torch.manual_seed(42)
        np.random.seed(42)

        params = {
            'mu': 1.81e-5,      
            'epsilon_p': 0.3, 
            'kappa': 1e-3,     
            'rho_f': 1.204,    
            'beta': 1.0/293.15,
            'T_ref': 293.15, 
            'C_p_f': 1005.0, 
            'k_eff': 6, 
            'g': 9.81, 
            'T_h': 353.15, 
            'T_c': 293.15, 
        }


        # Initialize the model
        normalizer = BrinkmanNormalization(params)
        model = PINNModel(normalizer, layers=layer_info)
        trainer = PINNTrainer(model, normalizer)

        # Check if go ahead to training or to run just one epoch (to find the initial losses)
        if state.pinn_loss_categories:
            ADAM_EPOCHS = 100
            LBFGS_EPOCHS = 1000
            loss_weights = self.assign_loss_weights(state.pinn_loss_categories)
            trainer.loss_weights = loss_weights
            # Save training logs
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                trainer.train(
                    epochs_adam=ADAM_EPOCHS,
                    epochs_lbfgs=LBFGS_EPOCHS,
                    n_interior=4096,
                    n_boundary=1024,
                    lr=1e-3,
                    print_every=100,
                    plot_every=100
                )
            print("\n" + "="*50 + "\nVALIDATION & FINAL ERROR\n" + "="*50)
            # try:
            checkpoint = torch.load('./results/best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model from epoch {checkpoint['epoch']} loaded (Loss: {checkpoint['loss']:.4e})")
            trainer.validate()
            _, mean_rel_error = trainer.compute_l2_error(state.validation_data)
            print("\nGenerating final solution plots from the best model...")
            trainer.plot_solution_fields(epoch=checkpoint['epoch'], final_epoch=True)
            print("Final plots saved successfully.")
            # except FileNotFoundError:
            #     print("Could not find 'best_model.pth'. Skipping validation and final plots.")
            training_logs = buffer.getvalue()
            models_history = state.models_history
            models_history.append(state.pinn_architecture)
            return {
                "project_status": "Send model error",
                "models_history": models_history,
                "pinn_training_logs": training_logs,
                "pinn_relative_error": mean_rel_error,
                "pinn_weights": "./results/best_model.pth"
            }
         
        else:
            ADAM_EPOCHS = 1
            LBFGS_EPOCHS = 0
            loss_weights = {
                'heat': 1.0,
                'momentum_y': 1.0,
                'momentum_x': 1.0,
                'bc_T': 1.0,
                'bc_u': 1.0,
                'bc_p': 1.0,
                'bc_neumann': 1.0
            }
            trainer.loss_weights = loss_weights
            initial_losses = trainer.train(
                                epochs_adam=ADAM_EPOCHS,
                                epochs_lbfgs=LBFGS_EPOCHS,
                                n_interior=4096,
                                n_boundary=1024,
                                lr=1e-3,
                                print_every=100,
                                plot_every=100)
            
            return {
                "project_status": "Send first epoch info",
                "initial_losses": initial_losses
            }

        
