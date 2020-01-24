import torch
import numpy as np

class SNN(torch.nn.Module):
    
    def __init__(self, layers):
        
        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        loss_seq = []
        
        for l in self.layers:
            x, loss = l(x)
            loss_seq.append(loss)
            
        return x, loss_seq
    
    
    def clamp(self):
        
        for l in self.layers:
            l.clamp()  
            
            
            
    def reset_parameters(self):
        
        for l in self.layers:
            l.reset_parameters()
            
            
class SpikingDenseLayer(torch.nn.Module):
    
    def __init__(self, nb_inputs, nb_outputs, spike_fn, w_init_mean, w_init_std, eps=1e-8):
        
        super(SpikingDenseLayer, self).__init__()
        
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.spike_fn = spike_fn
        self.eps = eps
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        self.w = torch.nn.Parameter(torch.empty((nb_inputs, nb_outputs)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(nb_outputs), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        h = torch.einsum("abc,cd->abd", x, self.w)
        
        nb_steps = h.shape[1]
        
        # membrane potential 
        mem = torch.zeros((batch_size, self.nb_outputs),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.nb_outputs),  dtype=x.dtype, device=x.device)
        
        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.nb_outputs),  dtype=x.dtype, device=x.device)
        
        d = torch.einsum("ab, ac -> bc", self.w, self.w)
            
        norm = (self.w**2).sum(0)
        
        
        for t in range(nb_steps):
            
            # reset term
            rst = torch.einsum("ab,bc ->ac", spk, d)
       
            # membrane potential update
            mem = (mem-rst)*self.beta + h[:,t,:]*(1.-self.beta)
            mthr = torch.einsum("ab,b->ab",mem, 1./(norm+self.eps))-self.b 
                
            spk = self.spike_fn(mthr)
            
            spk_rec[:,t,:] = spk   
            
        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()       
        
        loss = 0.5*(spk_rec**2).mean()
        
        return spk_rec, loss
   
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.nb_inputs))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)
    
    
class SpikingConvLayer(torch.nn.Module):
    
    def __init__(self, nb_inputs, nb_outputs,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std, eps=1e-8, stride=(1,1),flatten_output=False):
        
        super(SpikingConvLayer, self).__init__()
        
        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.spike_fn = spike_fn
        self.eps = eps
        
        self.flatten_output = flatten_output
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(np.ceil(((self.kernel_size-1)*self.dilation)/2).astype(int)),
                                      dilation=tuple(self.dilation),
                                      stride=tuple(self.stride))
        conv_x = conv_x[:,:,:,:self.nb_outputs]
        nb_steps = conv_x.shape[2]
        
        # membrane potential 
        mem = torch.zeros((batch_size, self.out_channels, self.nb_outputs),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, self.nb_outputs),  dtype=x.dtype, device=x.device)
        
        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, self.nb_outputs),  dtype=x.dtype, device=x.device)
        
        
        
        d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)     
        b = self.b.unsqueeze(1).repeat((1,self.nb_outputs))
            
        norm = (self.w**2).sum((1,2,3))
        
        
        for t in range(nb_steps):
            
            # reset term
            rst = torch.einsum("abc,bd ->adc", spk, d)
            
            # membrane potential update
            mem = (mem-rst)*self.beta + conv_x[:,:,t,:]*(1.-self.beta)
            mthr = torch.einsum("abc,b->abc",mem, 1./(norm+self.eps))-b 
                
            spk = self.spike_fn(mthr)
            
            spk_rec[:,:,t,:] = spk   
            
        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()       
        
        loss = 0.5*(spk_rec**2).mean()
        
        if self.flatten_output:
            
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels*self.nb_outputs)
            
        else:
            
            output = spk_rec
        
        return output, loss
   
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)
        
        
        
class ReadoutLayer(torch.nn.Module):
    
    "Fully connected readout"
    
    def __init__(self,  nb_inputs, nb_outputs, w_init_mean, w_init_std, eps=1e-8, time_reduction="mean"):
        
        
        assert time_reduction in ["mean", "max"], 'time_reduction should be "mean" or "max"'
        
        super(ReadoutLayer, self).__init__()
        

        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        
        self.eps = eps
        self.time_reduction = time_reduction
        
        
        self.w = torch.nn.Parameter(torch.empty((nb_inputs, nb_outputs)), requires_grad=True)
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.tensor(0.7*np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(nb_outputs), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()
        
        self.mem_rec_hist = None
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
       
        h = torch.einsum("abc,cd->abd", x, self.w)
        
        norm = (self.w**2).sum(0)
        
        if self.time_reduction == "max":
            nb_steps = x.shape[1]
            # membrane potential 
            mem = torch.zeros((batch_size, self.nb_outputs),  dtype=x.dtype, device=x.device)

            # memrane potential recording
            mem_rec = torch.zeros((batch_size, nb_steps, self.nb_outputs),  dtype=x.dtype, device=x.device)

            for t in range(nb_steps):

                # membrane potential update
                mem = mem*self.beta + (1-self.beta)*h[:,t,:]
                mem_rec[:,t,:] = mem
                
            output = torch.max(mem_rec, 1)[0]/(norm+1e-8) - self.b
            
        elif self.time_reduction == "mean":
            
            mem_rec = h
            output = torch.mean(mem_rec, 1)/(norm+1e-8) - self.b
        
        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()
        
        loss = None
        
        return output, loss
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean,
                              std=self.w_init_std*np.sqrt(1./(self.nb_inputs)))
        
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        if self.time_reduction == "max":
            self.beta.data.clamp_(0.,1.)
        
        
class SurrogateHeaviside(torch.autograd.Function):
    
    # Activation function with surrogate gradient
    sigma = 10.0

    @staticmethod 
    def forward(ctx, input):
        
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input*torch.sigmoid(SurrogateHeaviside.sigma*input)*torch.sigmoid(-SurrogateHeaviside.sigma*input)
        return grad
    