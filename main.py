"""
LENP model with condensation

"A Lumry-Eyring Nucleated-Polymerization (LENP) Model of
Protein Aggregation Kinetics 2. Competing Growth via
Condensation- and Chain-Polymerization"
Yi Li and Christopher Roberts, J Phys Chem B, 2009

by:
Pouria Akbari Mistani
p.a.mistani@gmail.com

"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import os
import pdb

class LENP(object):
    def __init__(self, x, delta, beta_gn, beta_cg, n_star, tf, k_type='CONSTANT'):
        self.x = x
        self.delta = delta
        self.beta_gn = beta_gn
        self.beta_cg = beta_cg
        self.n_star = n_star
        self.tfinal = tf
        self.maxstep = 5e-4
        self.num_timesteps_record = 1000
        self.t_evaluation = np.linspace(0, self.tfinal, self.num_timesteps_record)
        self.k_type = k_type

        if k_type=='CONSTANT':
            self.Kappa = 1.0


    def kappa(self, i, j):
        '''
        x <= i,j <= n_star - 1
        '''
        if i is None:
            I = np.arange(self.x, self.n_star)
            return 0.25*(1./I + 1./j)*(I + j)
        if j is None:
            J = np.arange(self.x, self.n_star)
            return 0.25*(1./i + 1./J)*(i + J)
        else:
            return 0.25*(1./i + 1./j)*(i + j)


    def model(self, t, y):
        m = y[0]
        a = y[1:]    # a[0] = a_x, a[1] = a_{x+1}, ...
        sigma = np.sum(a)
        out = []

        mdot = -self.x*m**self.x - self.delta*self.beta_gn*sigma*m**self.delta
        if self.k_type=='CONSTANT':
            yxdot= m**self.x - self.beta_gn*a[0]*m**self.delta - self.beta_gn*self.beta_cg*self.Kappa*a[0]*(a[0] + self.Kappa*np.sum(a) )
        else:
            yxdot= m**self.x - self.beta_gn*a[0]*m**self.delta - self.beta_gn*self.beta_cg*self.kappa(self.x, self.x)*a[0]*(a[0] + self.kappa(self.x, None).dot(a))
        out = [mdot, yxdot]

        for i in range( self.x+1, self.n_star):
            indx = i - self.x
            if self.k_type=='CONSTANT':
                tmp  = self.beta_gn*(a[indx - self.delta] - a[indx])*m**self.delta
                tmp -= self.beta_gn*self.beta_cg*a[indx]*(self.Kappa*a[indx] + self.Kappa*np.sum(a))
                tmp += self.beta_gn*self.beta_cg*np.sum( [ self.Kappa*a[i - j]*a[j - self.x] for j in range(self.x, int(i/2)) ] )
                # if self.n_star>=self.x/2:
                #     nstar = self.x/2
                #     while nstar <= self.n_star:
                #         for ind1 in range(self.x, nstar-1):
                #             ind2 = nstar - 1 - ind1
                #             if ind1 >= ind2:
                #                 tmp += self.beta_gn*self.beta_cg*self.Kappa*a[ind1]*a[ind2]
                #         nstar += 1


            else:
                tmp  = self.beta_gn*(a[indx - self.delta] - a[indx])*m**self.delta
                tmp -= self.beta_gn*self.beta_cg*a[indx]*(self.kappa(i, i)*a[indx] + self.kappa(i, None).dot(a))
                tmp += self.beta_gn*self.beta_cg*np.sum( [ self.kappa(i - j, j)*a[i - j]*a[j - self.x] for j in range(self.x, int(i/2)) ] )
            out.append(tmp)
        return out


    def solve(self, SOLVER='LSODA'):
        y0 = np.zeros(self.n_star - self.x + 1)
        y0[0] = 1.0
        self.sol = solve_ivp(self.model, [0, self.tfinal], y0, method=SOLVER, max_step=self.maxstep, rtol=1e-3, t_eval=self.t_evaluation)
        self.t_sol = self.sol.t
        self.m_sol = self.sol.y[0]
        self.As_sol = self.sol.y[1:]


    def get_colors(self, num, cmmp='coolwarm'):
        cmap = plt.cm.get_cmap(cmmp)
        cs = np.linspace(0, 1, num)
        colors = []
        for i in range(num):
            colors.append(cmap(cs[i]))
        return np.array(colors)

    def get_statistics(self):
        js = np.arange(self.x, self.n_star)
        self.sigma = np.sum(self.As_sol, axis=0)
        self.Lambda_1 = np.array([aj.dot(js) for aj in self.As_sol.T])
        self.Lambda_2 = np.array([aj.dot(js**2) for aj in self.As_sol.T])
        self.MwAgg_Mmon  = self.Lambda_2/self.Lambda_1
        self.MnAgg_Mmon  = self.Lambda_1/self.sigma
        self.MwAgg_MnAgg = self.sigma*self.Lambda_2/self.Lambda_1**2

    def plot_sols(self, save_add):
        num_a_plots = 10
        cols = self.get_colors(num_a_plots+1)
        try:
            os.mkdir(save_add)
        except OSError:
            print ("Creation of the %s directory failed" %save_add)
        else:
            print ("Successfully created the Figures directory %s" %save_add)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        counter = 0
        for i in range(0, self.n_star - self.x, max((self.n_star-self.x)/num_a_plots, 1)):
            ax.plot(self.t_sol, self.As_sol[i]*1e4, lw=2, color=cols[counter], label='i='+ str(i+self.x))
            counter += 1
        ax.set_xlabel(r'$\rm \theta$', fontsize=25)
        ax.set_ylabel(r'$\rm a_i\times 10^4$', fontsize=25)
        ax.legend(fontsize=15, frameon=False)
        plt.savefig(save_add + "/ai_theta.png")

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        counter = 0
        for j in range(0, self.num_timesteps_record, int(max(self.num_timesteps_record/num_a_plots, 1))) :
            ax.plot(range(self.x, self.n_star), self.As_sol.T[j]*1e4, lw=2, color=cols[counter], label=r'$\theta=$'+ str(self.t_evaluation[j])[:4])
            counter += 1
        ax.set_xlabel(r'$\rm j$', fontsize=25)
        ax.set_ylabel(r'$\rm a_j\times 10^4$', fontsize=25)
        ax.legend(fontsize=15, frameon=False)
        plt.savefig(save_add + "/ai_i.png")

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.plot(self.t_sol, self.m_sol, lw=2, color='k')
        ax.set_xlabel(r'$\rm \theta$', fontsize=25)
        ax.set_ylabel(r'$\rm m$', fontsize=25)
        ax.set_ylim([0,1.1])
        plt.savefig(save_add + "/m_theta.png")

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.plot(self.t_sol, np.log10(self.sigma), lw=2, color='k')
        ax.set_xlabel(r'$\rm \theta$', fontsize=25)
        ax.set_ylabel(r'$\rm \log_{10}\sigma$', fontsize=25)
        plt.savefig(save_add + "/sigma_theta.png")

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.plot(1 - self.m_sol, self.MwAgg_Mmon, lw=2, color='k')
        ax.set_xlabel(r'$\rm 1 - m$', fontsize=25)
        ax.set_ylabel(r'$\rm M_w^{agg} / M_{mon}$', fontsize=25)
        plt.savefig(save_add + "/MwAggMmon_1minusM.png")

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.plot(1 - self.m_sol, self.MwAgg_MnAgg - 1, lw=2, color='k')
        ax.set_xlabel(r'$\rm 1 - m$', fontsize=25)
        ax.set_ylabel(r'$\rm M_w^{agg}/ M_{n}^{agg} - 1$', fontsize=25)
        plt.savefig(save_add + "/MwAggMnAgg_1minusM.png")





case = 'Ib'

#################################
if case=='exploratory':
    x = 6
    delta = 1
    beta_gn = 100
    beta_cg = 1
    n_star = 100
    tfinal = 1.0
    solverType = 'LSODA'
    reactionType='CONSTANT'



#################################
# figure 5 of paper comparisons
if case=='A':
    x = 6
    delta = 1
    beta_gn = 1000
    n_star = 400
    tfinal = 1
    reactionType='DIFFUSION'
    solverType = 'BDF'
    beta_cg = 0.0
elif case=='B':
    x = 6
    delta = 1
    beta_gn = 1000
    n_star = 400
    tfinal = 0.4
    reactionType='DIFFUSION'
    solverType = 'BDF'
    beta_cg = 10.0
elif case=='C':
    x = 6
    delta = 1
    beta_gn = 1000
    n_star = 400
    tfinal = 1
    reactionType='DIFFUSION'
    solverType = 'BDF'
    beta_cg = 1e-4
elif case=='D':
    x = 6
    delta = 1
    beta_gn = 1000
    n_star = 400
    tfinal = 1
    reactionType='DIFFUSION'
    solverType = 'BDF'
    beta_cg = 20.0


#################################
# different types of solutions
if case=='Ia':
    x = 6
    delta = 1
    beta_gn = 1000
    beta_cg = 10
    n_star = 400
    tfinal = 1.0
    solverType = 'BDF'
    reactionType='CONSTANT'

if case=='Ib':
    x = 6
    delta = 1
    beta_gn = 1000
    beta_cg = 10
    n_star = 10
    tfinal = 1
    solverType = 'BDF'
    reactionType='CONSTANT'

if case=='Ic':
    x = 6
    delta = 1
    beta_gn = 0.1
    beta_cg = 0
    n_star = 400
    tfinal = 1
    solverType = 'BDF'
    reactionType='CONSTANT'

if case=='II':
    x = 6
    delta = 1
    beta_gn = 1000
    beta_cg = 0.05
    n_star = 400
    tfinal = 0.75
    solverType = 'BDF'
    reactionType='CONSTANT'

if case=='IVa':
    x = 6
    delta = 1
    beta_gn = 1000
    beta_cg = 0.5
    n_star = 400
    tfinal = 1
    solverType = 'BDF'
    reactionType='CONSTANT'

#################################

save_add = './Figures/'+case

lp = LENP(x, delta, beta_gn, beta_cg, n_star, tfinal, reactionType)
lp.solve(solverType)
lp.get_statistics()
lp.plot_sols(save_add)
