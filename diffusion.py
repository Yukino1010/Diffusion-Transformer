import torch
import torch.nn as nn
from tqdm import tqdm 

def CFG(model, z, t, y, n_classes, cfg_scale = 4.0):
    n = z.shape[0]
    t = torch.cat([t, t], dim=0) 
    z_combined = torch.cat([z, z], dim=0)  

    y_null = torch.full((n,), fill_value=n_classes, device=z.device)
    y_combined = torch.cat([y, y_null], dim=0)
    model_out = model(z_combined, t, y_combined)

    eps, rest = model_out[:, :3], model_out[:, 3:] 
    cond_eps, uncond_eps = torch.split(eps, n, dim=0)

    # eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) 
    eps = torch.cat([half_eps, half_eps], dim=0)  
    eps, _ = torch.cat([eps, rest], dim=1).chunk(2, dim=0) # (n, 4, H, W)
    return eps

class GaussianDiffusion(nn.Module):
    def __init__(self, device, timestep=1000):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, timestep, dtype=torch.float32, device=device)

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # posterior q(x_t-1 | x_t, x_0)
        self.posterior_variance = self.beta * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.log_posterior_variance = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.mean_coef1 = self.beta * self.alpha_bar_prev**0.5 / (1. - self.alpha_bar)
        self.mean_coef2 = (1. - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1. - self.alpha_bar)

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip = torch.sqrt(1. / self.alpha_bar[t])[:, None, None, None]
        sqrt_recipm1 = torch.sqrt(1. / self.alpha_bar[t] - 1)[:, None, None, None]
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def predict_noise_from_start(self, x_t, t, x0):
        sqrt_recip = torch.sqrt(1. / self.alpha_bar[t])[:, None, None, None]
        sqrt_recipm1 = torch.sqrt(1. / self.alpha_bar[t] - 1)[:, None, None, None]
        return (sqrt_recip * x_t - x0) / sqrt_recipm1
    
    def q_sample(self, x_0, t, noise=None):
        # sample x_t from q(x_t | x_0)
        # x_t = x_0 * alpha_bar_sqrt + one_minus_alpha_bar * noise 
        if noise is None:
            noise = torch.randn_like(x_0)
    
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, x_t, pred_noise, t):
        # sample x_t-1 from p(x_t-1 | x_t)
        # p(x_{t-1} | x_t) = N(mean, variance)
        # coef1 = beta[t] ＊ alpha_bar[t-1] ** 0.5  / (1 - alpha_bar[t])
        # coef2 = (1 - alpha_bar[t-1]) * alpha[t] ** 0.5 / (1 - alpha_bar[t])
        # - - - - - - 
        # mean = coef1 * x_0 + coef2 * x_t
        # variance = beta[t] * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])

        coef1 = self.mean_coef1[t][:, None, None, None]
        coef2 = self.mean_coef2[t][:, None, None, None]
        log_var = self.log_posterior_variance[t][:, None, None, None]
        x_0 = self.predict_start_from_noise(x_t, t, pred_noise).clamp(-1, 1)

        mean = x_0 * coef1 + x_t * coef2
        noise = torch.randn_like(x_t) if t[0] > 0 else 0.
        pred_img = mean + (0.5 * log_var).exp() * noise
        return pred_img, x_0
    
    def ddim_sample(self, model, cond, n_classes, \
                    eta=0, timesteps=1000, sampling_timesteps=200):
        model.eval()
        cond = cond.to(self.device)
        times = torch.linspace(-1, timesteps - 1, steps = sampling_timesteps + 1)  
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_t = torch.randn([cond.size(0), 4, 32, 32], device = self.device)
        x_0 = None

        for time, time_next in tqdm(time_pairs, desc = 'Sampling loop time step'):
            t = torch.full((cond.size(0),), time, device = self.device, dtype = torch.long)
            pred = CFG(model, x_t, t, cond, n_classes)
            x_0 = self.predict_start_from_noise(x_t, t, pred)
            pred_noise = self.predict_noise_from_start(x_t, t, x_0)
            
            if time_next < 0:
                x_t = x_0
                continue
            
            alpha = self.alpha_bar[time]
            alpha_next = self.alpha_bar[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x_t)
            x_t = x_0 * alpha_next.sqrt() + c * pred_noise + sigma * noise
        model.train()
        return x_t