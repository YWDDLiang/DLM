import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from symmbfn.model.cspnet.cspnet import CSPNet
import numpy as np
#from absl import logging
from symmbfn.common.data_utils import sg_to_ks_mask, mask_ks
import math
import hydra
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PI = math.pi
SYM_GROUPS = 13


class bfnBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)    

    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x +  torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma    

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        x: [N, K]
        """
        beta = beta1 * (t**2)  # (B,)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1).view(-1) * loss.view(-1) * torch.pow(sigma1, -2 * t).view(-1)
    
    def ctime4continuous_cyclic_loss(self, t, sigma1, x_pred, x):
        fst_diff = (x_pred - x) % 1.0
        alt_diff = (x - x_pred) % 1.0
        diff = torch.where(fst_diff < alt_diff, fst_diff, alt_diff)
        loss = diff.view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1).view(-1) * loss.view(-1) * torch.pow(sigma1, -2 * t).view(-1)

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x):
        # TODO not debuged yet
        weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        return weight * (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K):
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def dtime_discrete_loss(self):
        pass

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class bfn4DNG(bfnBase):
    def __init__(
        self,
        dynamics,
        beta1,
        beta1_sym,
        smooth = True,
        pred_type = True,
        max_atoms = 100,
        max_axes=15,
        max_syms=13,    
        device=DEVICE,
        sigma1_coord=0.02,
        sigma1_lattice=0.02,
        sample_steps=500,
        t_min=0.0001,
    ):
        super(bfn4DNG, self).__init__()
        self.max_atoms = max_atoms
        self.max_axes = max_axes
        self.max_syms = max_syms
        self.cspnet = dynamics
        
        self.device = device
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32, device=self.device)
        self.sigma1_lattice = torch.tensor(sigma1_lattice, dtype=torch.float32, device=self.device)
        self.beta1 = torch.tensor(beta1, dtype=torch.float32, device=self.device)
        self.beta1_sym = torch.tensor(beta1_sym, dtype=torch.float32, device=self.device)
        self.sample_steps = sample_steps
        self.t_min = t_min
       

    def interdependency_modeling(
        self,
        time,
        mu_pos_t,
        gamma_coord,
        theta_type,
        mu_lattice_t,
        gamma_lattice,
        num_atoms,
        theta_sym,
        spacegroup,
        segment_ids=None,
        prop=None,
        inference=False,
    ):
        """
        Args:
            time: should be a scalar tensor or the shape of [node_num x batch_size, 1]
            h_state: [node_num x batch_size, max_atoms]
            coord_state: [node_num x batch_size, 3]
        """
       
        lattice_final, coord_final, k_hat, sym_out = self.cspnet(time, theta_type, mu_pos_t, mu_lattice_t, num_atoms, segment_ids, theta_sym,spacegroup, prop)


        eps_lattice_pred = lattice_final #+ mu_lattice_t
        eps_lattice_pred = torch.clamp(eps_lattice_pred, min=-10, max=10)
        lattice_pred = (
            mu_lattice_t / gamma_lattice
            - torch.sqrt((1 - gamma_lattice) / gamma_lattice) * eps_lattice_pred
        )
        ks_mask, ks_add = sg_to_ks_mask(spacegroup)
        lattice_pred = mask_ks(lattice_pred, ks_mask, ks_add)

        eps_coord_pred = coord_final + mu_pos_t
      

        eps_coord_pred = torch.clamp(eps_coord_pred, min=-10, max=10)
        
       
        coord_pred = (
            mu_pos_t / gamma_coord
            - torch.sqrt((1 - gamma_coord) / gamma_coord) * eps_coord_pred
        )
        coord_pred = coord_pred % 1.0
        #coord_pred = torch.clamp(coord_pred, 0, 1)
        sym_out = sym_out.view((-1, self.max_syms))
        return coord_pred, k_hat, lattice_pred, sym_out
    

    def loss_one_step(
        self,
        t,
        x,
        pos,
        ks,
        num_atoms,
        spacegroup,
        site_syms,
        segment_ids=None,
        prop=None,
    ):  
        ks = ks.view(-1,6)
        types = torch.nn.functional.one_hot(x, num_classes=self.max_atoms)
        site_syms = site_syms.view((-1, site_syms.size()[-1]))
        #type = torch.unsqueeze(x, dim=-1)
        mask = t > self.t_min
        # bayesian update
        atom_time = t.index_select(0, segment_ids)
        atom_mask = atom_time > self.t_min
        t = torch.clamp(t, min=self.t_min)
        atom_time = torch.clamp(atom_time, min=self.t_min)

        theta_type = self.discrete_var_bayesian_update(atom_time, self.beta1, types, self.max_atoms)
        theta_sym = self.discrete_var_bayesian_update(atom_time.repeat_interleave(self.max_axes).unsqueeze(-1), self.beta1_sym, site_syms, self.max_syms)
        #sigma1_coord = self.sigma1_coord / (num_atoms.index_select(0, segment_ids).view(-1, 1).float()**(1/3))

        #mu_coord, gamma_coord = self.continuous_cyclic_var_bayesian_update(
        #    atom_time, sigma1=self.sigma1_coord, x=pos
        #) #new
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
           atom_time, sigma1=self.sigma1_coord, x=pos
        )
        mu_coord = mu_coord %1.0
        mu_lattice, gamma_lattice = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_lattice, x=ks
        )
        
        ks_mask, ks_add = sg_to_ks_mask(spacegroup)
        mu_lattice = mask_ks(mu_lattice, ks_mask, ks_add)
      
        mu_coord = torch.where(atom_mask, mu_coord, torch.zeros_like(mu_coord))
        mu_lattice = torch.where(mask, mu_lattice, torch.zeros_like(mu_lattice))

        coord_pred, k_hat, lattice_pred, sym_out = self.interdependency_modeling(
            t,
            mu_pos_t=mu_coord,
            gamma_coord=gamma_coord,
            theta_type=theta_type,
            mu_lattice_t = mu_lattice,
            gamma_lattice=gamma_lattice,
            num_atoms= num_atoms,
            segment_ids=segment_ids,
            theta_sym=theta_sym,
            spacegroup=spacegroup,
            prop=prop
        )

        posloss = self.ctime4continuous_cyclic_loss(
            t=atom_time, sigma1=self.sigma1_coord, x_pred=coord_pred, x=pos
        )
        latticeloss = self.ctime4continuous_loss(
            t=t, sigma1=self.sigma1_lattice, x_pred=lattice_pred, x=ks
        )
        type_loss = self.ctime4discrete_loss(t=atom_time, beta1=self.beta1, one_hot_x=types, p_0=k_hat, K=self.max_atoms)
        
        sym_loss = self.ctime4discrete_loss(t=atom_time.repeat_interleave(self.max_axes), beta1=self.beta1_sym, one_hot_x=site_syms, p_0=sym_out, K=self.max_syms)

        return (
            posloss,
            type_loss,
            latticeloss,
            sym_loss,
            (mu_coord, k_hat, coord_pred, gamma_coord),
        )

    def forward(
        self, n_nodes, num_atoms, num_crystals, spacegroup, prop, sample_steps=None, segment_ids=None
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1, note here we use a single t
            theta_t: [node_num x batch_size, atom_type]
            mu_t: [node_num x batch_size, 3]
        """
        mu_pos_t = torch.zeros((n_nodes, 3)).to(self.device)  # [N, 4] coordinates prior
        mu_lattice_t = torch.zeros((num_crystals,6)).to(self.device)

        ro_coord = torch.tensor(1, dtype=torch.float32).to(self.device)
        ro_lattice = torch.tensor(1, dtype=torch.float32).to(self.device)

        theta_type = torch.ones((n_nodes, self.max_atoms)).to(self.device) * (1/self.max_atoms)
        theta_sym = torch.ones((n_nodes * self.max_axes, self.max_syms)).to(self.device) * (1/self.max_syms)

        if sample_steps is None:
            sample_steps = self.sample_steps
        theta_traj = []
        for i in range(1, sample_steps + 1):
            t = torch.ones((num_crystals, 1)).to(self.device) * (i - 1) / sample_steps
            t = torch.clamp(t, min=self.t_min)
            atom_time = t.index_select(0, segment_ids)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * atom_time)
            gamma_lattice = 1 - torch.pow(self.sigma1_lattice, 2 * t)

            coord_pred, k_hat, lattice_pred, sym_out = self.interdependency_modeling(
            time = t,
            mu_pos_t=mu_pos_t,
            gamma_coord=gamma_coord,
            theta_type=theta_type,
            mu_lattice_t = mu_lattice_t,
            gamma_lattice=gamma_lattice,
            num_atoms= num_atoms,
            theta_sym=theta_sym,
            spacegroup=spacegroup,
            segment_ids=segment_ids,
            prop=prop,
            inference=True
            )
            #-------coords
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )
            y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(
                1 / alpha_coord
            )
            #y_coord = y_coord % 1.0
            mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
                ro_coord + alpha_coord
            )
            mu_pos_t = mu_pos_t % 1.0
            ro_coord = ro_coord + alpha_coord
            #----------------lattice
            alpha_lattice = torch.pow(self.sigma1_lattice, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_lattice, 2 / sample_steps)
            )
            y_lattice = lattice_pred + torch.randn_like(lattice_pred) * torch.sqrt(
                1 / alpha_lattice
            )
            mu_lattice_t = (ro_lattice * mu_lattice_t + alpha_lattice * y_lattice) / (
                ro_lattice + alpha_lattice
            )

            ks_mask, ks_add = sg_to_ks_mask(spacegroup)
            mu_lattice_t = mask_ks(mu_lattice_t, ks_mask, ks_add)
            ro_lattice = ro_lattice + alpha_lattice
            
            #------------types
            alpha_type = self.beta1 * ((2 * i - 1)/sample_steps**2)
            e_k = F.one_hot(torch.multinomial(k_hat, num_samples=1), self.max_atoms).squeeze()
            mu = alpha_type * (self.max_atoms * e_k - 1)
            y_type = mu + torch.randn_like(k_hat) * torch.sqrt(alpha_type * self.max_atoms)
            theta_type_dash = torch.exp(y_type) * theta_type

            theta_type = theta_type_dash / theta_type_dash.sum(dim=1).unsqueeze(1)

            #----------syms
            alpha_sym = self.beta1_sym * ((2 * i - 1)/sample_steps**2)
            es_k = F.one_hot(torch.multinomial(sym_out, num_samples=1), self.max_syms).squeeze()
            mu_syms = alpha_sym * (self.max_syms * es_k - 1)
            y_syms = mu_syms + torch.randn_like(sym_out) * torch.sqrt(alpha_sym * self.max_syms)
            theta_sym_dash = torch.exp(y_syms) * theta_sym

            theta_sym = theta_sym_dash / theta_sym_dash.sum(dim=1).unsqueeze(1)

        mu_pos_final, k_hat_final, mu_lattice_final, sym_final_out = self.interdependency_modeling(
            time=torch.ones((num_crystals, 1)).to(self.device),
            mu_pos_t=mu_pos_t,
            gamma_coord=1 - self.sigma1_coord**2,
            theta_type=theta_type,
            mu_lattice_t=mu_lattice_t,
            gamma_lattice=1 - self.sigma1_lattice**2,
            num_atoms=num_atoms,
            theta_sym=theta_sym,
            spacegroup=spacegroup,
            segment_ids=segment_ids,
            prop=prop,
            inference=True
        )
        
        type_final = F.one_hot(torch.multinomial(k_hat_final, num_samples=1), self.max_atoms).squeeze()
        sym_final = F.one_hot(torch.multinomial(sym_final_out, num_samples=1), self.max_syms).squeeze()
        sym_final = sym_final.view(-1, 15,13)
        theta_traj.append((mu_pos_final, type_final, mu_lattice_final, sym_final))
        return theta_traj
