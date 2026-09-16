"""Torch-free CPU export for the isolated PhysMix comparison, not production."""
from __future__ import annotations

import numpy as np
from scipy.special import erf,expit


def radial_acceleration(position,mass,charge,radius,activity,attraction,coupling,well):
    """Exact radial sum via a weighted graph Laplacian, without N*N*D arrays.

    Sum_j c_ij (r_i-r_j) = r_i Sum_j c_ij - (C @ r)_i.
    Self terms are exactly zero and removed before the matrix contraction.
    The nonnegative distance guard removes only Gram roundoff below zero.
    """
    norm = (position**2).sum(-1)
    squared = np.maximum(norm[:,:,None]+norm[:,None,:]-2*(position@position.transpose(0,2,1)),0.)
    distance = np.sqrt(squared+.5**2)
    m,q,s = mass[...,0],charge[...,0],radius[...,0]
    power = ((s[:,:,None]+s[:,None,:])*.5/distance)**6
    radial = (-attraction*m[:,:,None]*m[:,None,:]+coupling*q[:,:,None]*q[:,None,:])/distance**2
    radial += 24*well/distance*(2*power**2-power)
    coefficients = activity[:,None,:]*radial/distance
    diagonal = np.arange(position.shape[1])
    coefficients[:,diagonal,diagonal] = 0.
    return (position*coefficients.sum(-1,keepdims=True)-coefficients@position)/mass


class NumpyPairComparison:
    def __init__(self,arrays,mode,steps=8):
        if mode not in ('current','capacity_control','physmix') or not 1<=steps<=32:
            raise ValueError('invalid comparison mode or relaxation count')
        self.arrays = {name:np.asarray(value,np.float32) for name,value in arrays.items()}
        if any(not np.isfinite(value).all() for value in self.arrays.values()):
            raise ValueError('nonfinite comparison weights')
        self.mode,self.steps = mode,steps

    def linear(self,x,name):
        a = self.arrays
        return x@a[name+'.weight'].T+a.get(name+'.bias',0.)

    def layer(self,x,name):
        a = self.arrays
        centered = x-x.mean(-1,keepdims=True)
        return centered/np.sqrt((centered**2).mean(-1,keepdims=True)+1e-5)*a[name+'.weight']+a[name+'.bias']

    @staticmethod
    def gelu(x):
        return .5*x*(1+erf(x/np.sqrt(2.)))

    def bvalue(self,name):
        return self.arrays['backbone.'+name.replace('.','__')]

    def blinear(self,x,name):
        return x@self.bvalue(name+'.weight').T+(self.bvalue(name+'.bias') if 'backbone.'+(name+'.bias').replace('.','__') in self.arrays else 0.)

    def aggregate(self,e,amounts,context):
        weights = amounts/amounts.sum(-1,keepdims=True)
        mean = (weights[...,None]*e).sum(1)
        variance = (weights[...,None]*(e-mean[:,None])**2).sum(1)
        ctx = np.maximum(0.,self.blinear(context,'context_in'))
        logits = (self.blinear(e,'attention_key')*self.blinear(ctx,'attention_query')[:,None]).sum(-1)/8
        logits = np.where(amounts>0,logits+np.log(np.maximum(amounts,1e-30)),-1e30)
        attention = np.exp(logits-logits.max(-1,keepdims=True))*(amounts>0)
        attention /= np.maximum(attention.sum(-1,keepdims=True),1e-30)
        attended = (attention[...,None]*e).sum(1)
        fused = np.concatenate((mean,variance,attended,ctx,np.zeros((len(e),96),np.float32)),-1)
        h = np.maximum(0.,self.blinear(np.maximum(0.,self.blinear(fused,'fusion')),'shared_in'))
        for i in range(2):
            name = f'shared.{i}'
            centered = h-h.mean(-1,keepdims=True)
            normed = centered/np.sqrt((centered**2).mean(-1,keepdims=True)+1e-5)
            normed = normed*self.bvalue(name+'.norm.weight')+self.bvalue(name+'.norm.bias')
            h = h+self.blinear(np.maximum(0.,self.blinear(normed,name+'.up')),name+'.down')
        return h,mean,variance

    def dynamics(self,e,amounts):
        prefix = 'interaction.'
        total = amounts.sum(-1,keepdims=True)
        relative = amounts/total
        activity = amounts/(1+total)
        h = self.layer(self.gelu(self.linear(self.layer(e,prefix+'encoder.0'),prefix+'encoder.1')),prefix+'encoder.3')
        position = .25*np.tanh(self.linear(h,prefix+'position'))
        initial = position.copy()
        velocity = .1*np.tanh(self.linear(h,prefix+'velocity'))
        mass = np.logaddexp(0.,self.linear(h,prefix+'mass'))+.1
        initial_mass = mass.copy()
        charge = np.tanh(self.linear(h,prefix+'charge'))
        radius = .1+.4*expit(self.linear(h,prefix+'radius'))
        attraction,coupling,well,speed_limit,decay = np.logaddexp(0.,self.arrays[prefix+'log_constants'])+.01
        dt = .1/self.steps
        displacement = np.zeros_like(mass)
        for _ in range(self.steps):
            acceleration = radial_acceleration(position,mass,charge,radius,activity,attraction,coupling,well)
            trial = velocity+dt*acceleration
            norm = np.maximum(np.linalg.norm(trial,axis=-1,keepdims=True),1e-12)
            velocity = speed_limit*np.tanh(norm/speed_limit)*trial/norm
            position = position+dt*velocity
            mass = mass*np.exp(-decay*dt/(mass**2+.5**2))
            displacement += ((position-initial)**2).sum(-1,keepdims=True)/self.steps
        vector = np.concatenate((position,displacement,(velocity**2).sum(-1,keepdims=True),mass,
            mass/initial_mass,np.abs(charge),np.broadcast_to(np.log1p(total)[:,None],(*amounts.shape,1))),-1)
        out = self.linear((relative[...,None]*vector).sum(1),prefix+'output.0')
        return self.linear(self.gelu(self.layer(out,prefix+'output.1')),prefix+'output.3')

    def encode(self,embedded,amounts,context):
        e,w,c = map(lambda x:np.asarray(x,np.float32),(embedded,amounts,context))
        if (e.ndim!=3 or e.shape[-1]!=256 or w.shape!=e.shape[:2] or c.shape!=(len(e),64)
            or any(not np.isfinite(x).all() for x in (e,w,c)) or np.any(w<0)
            or np.any(w.sum(-1)<=0) or not np.isfinite(w.sum(-1)).all()):
            raise ValueError('finite complete molecular mixtures required')
        h,mean,variance = self.aggregate(e,w,c)
        if self.mode!='current':
            extra = self.dynamics(e,w) if self.mode=='physmix' else self.linear(
                self.gelu(self.linear(np.concatenate((mean,variance),-1),'interaction.0')),'interaction.2')
            h = h+.25*np.tanh(self.arrays['gate'])*extra
        return h,np.log1p(w.sum(-1))

    def __call__(self,a,wa,b,wb,context):
        first,ca = self.encode(a,wa,context)
        second,cb = self.encode(b,wb,context)
        cosine = (first*second).sum(-1)/(np.maximum(np.linalg.norm(first,axis=-1),1e-8)*np.maximum(np.linalg.norm(second,axis=-1),1e-8))
        features = np.concatenate((np.abs(first-second),first*second,cosine[:,None],np.abs(ca-cb)[:,None],(ca*cb)[:,None]),-1)
        return expit(self.linear(self.gelu(self.linear(features,'head.0')),'head.3').squeeze(-1))
