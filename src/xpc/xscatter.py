#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 6 12:05:10 2023
Last updated on Thu Aug 5 12:37:00 2024

@author: gjadick

A script for retrieving complex index of refraction values
in the soft x-ray energy range.


#####################################################################
### EXAMPLE USAGE
#####################################################################

To get delta and beta factors for water as a function of energy:

```
energies = np.arange(1., 140., 1.)  # array of energy in keV
material = 'H(88.8)O(11.2)'         # water composition
density = 1.0                       # material density in g/cm^3

delta, beta = get_delta_beta_mix(matcomp, energies, density)
```


#####################################################################
### NOTES
##################################################################### 

This script relies on the EPDL97 database of Photon Anomalous 
Scattering Factors found at:
    https://www-nds.iaea.org/epdl97/
    
To access the database, each element's data has been saved here in 
a separate text file. As explained by IAEA, the file format is:

    The first line defines,
    1)  The Atomic Number of the Element (Z = 1 for hydrogen)
    2)  The Number of Tabulated Energy Points (361)
    3)  The Atomic Weight of the Naturally Occurring Element (1.008)
    4)  The STP Density in grams/cc (8.988e-5)
    5)  Definition of the Element in text (1-H -Nat)  
     
    The Second Line contains titles for each Column
    1)  MeV - the units of Energy
    2)  F1-Total - Total f1 (sum of ionization and excitation)
    3)  Z+F1 - In this form it is easier to see FF+F1 approaching zero at low energy.   
    4)  F1-Ionize - Contribution of Ionization to f1
    5)  F1-Excite - Contribution of Excitation to f1
    6)  F2-Total - Total f2 (sum of ionization and excitation)
    7)  F2-Ionize - Contribution of Ionization to f2
    8)  F2-Excite - Contribution of Excitation to f2
    9)  Coherent (barns) - Coherent Cross Section defined by integrating the above equation.

As validation of this script, it's useful to know that the package 
`periodictable` includes a method for getting scatter factors. However, it uses
a different database, which is generally considered less accurate at >30 keV.
    B. L. Henke, E. M. Gullikson, and J. C. Davis. “X-ray interactions: 
    photoabsorption, scattering, transmission, and reflection at 
    E=50-30000 eV, Z=1-92”, Atomic Data and Nuclear Data Tables 54 no.2, 
    181-342 (July 1993).
    
In fact, the periodictable package doesn't provide any values above 30 keV. 
As a check, I confirmed that the Henke 1993 values match the EPDL97-based 
values computed in this script reasonably well at the available energies. 

For reference, example usage of periodictable to get delta, beta:
    ```
    import periodictable as pt 
    import periodictable.xsf
    
    # Example material is PMMA -- this is the xscatter formatting:
    matcomp = 'H(8.0541)C(59.9846)O(31.9613)' 
    density =  1.19  # [g/cm^3]
    
    # To use periodictable, define PMMA in this format:
    elem = periodictable.formula('8.0541%wt H // 59.9846% C // O')
    elem.density = density 

    # Get the complex index of refraction:
    n = pt.xsf.index_of_refraction(elem, energy=my_energy_values)
    delta = 1 - n.real
    beta = -n.imag
    ```
    
"""

import os
import re
import numpy as np
import jax.numpy as jnp
from pathlib import Path

rootpath = Path(__file__).parent / 'xscatter_data'


### CONSTANTS
r_e = 2.8179403262e-15        # classical electron radius, m
N_A = 6.02214076e+23          # Avogadro's number, num/mol
pi = np.pi
h  = 6.62607015e-34           # Planck constant, J/Hz
c = 299792458.0               # speed of light, m/s
J_eV = 1.602176565e-19        # J per eV conversion

elements =  ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
    'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',\
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',\
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']


def get_wavelen(energy):
    """energy in keV -> returns wavelength in m"""
    try:
        len(energy)
        energy = np.array(energy)
    except:
        pass  
    return 1e-3*h*c / (energy*J_eV)


def get_wavenum(energy):
    """energy in keV -> returns wavenum in m^-1"""
    try:
        len(energy)
        energy = np.array(energy)
    except:
        pass  
    return 2*np.pi / get_wavelen(energy)


def get_energy(wavelen):
    """wavelen in m --> returns energy in keV"""
    try:
        len(wavelen)
        wavelen = np.array(wavelen)
    except:
        pass  
    return 1e-3*h*c / (wavelen*J_eV)


def get_filename(elem):
    """
    Get the filename with data for a given element.

    INPUTS:
    elem - short element name (str)
           e.g. for Hydrogen, elem='H'

    OUTPUTS:
    filename - the path to the data file
    """
    Z = elements.index(elem)+1
    filename =  str(rootpath / f'za{int(Z):03}000.txt')
    return filename


def to_float(s):
    '''
    Convert an EPDL97-formatted number to a float.
    Their scientific notation format for "X * 10^-Y" 
    is "X-Y", or for "X * 10+Y" it is "X+Y", 
    neither of which can convert to float using 
    Python type casting alone.
    
    Perhaps this function is excessively convoluted;
    I welcome simpler solutions.
    '''
    try:    # attempt type casting
        result = float(s)
    except: # convert string
        # check for negative val
        sign = 1
        if s[0]=='-':
            sign = -1 
            s = s[1:]
        # get power
        if len(s[1:].split('+'))==2:
            split_char = '+'
            pow_sign = 1.0
        elif len(s[1:].split('-'))==2:
            split_char = '-'
            pow_sign = -1.0
        else:
            print('FLOAT CONVERSION ERROR', s)
            return None
        # calc float
        A = float(s.split(split_char)[0])
        pow = pow_sign * float(s.split(split_char)[1])
        result = sign * A * 10**pow
        
    return result


def get_amass(elem):
    """
    Finds the atomic mass of an element in g/mole.

    INPUTS:
    elem - short element name (str)
           e.g. for Hydrogen, elem='H'
    
    OUTPUTS:
    amass - double specifiying atomic mass in grams per mole
    """
    with open(get_filename(elem)) as f:
        amass = to_float(f.readline().split()[2])
    return amass


def get_f1_f2(elem, energy):
    """
    Get energy-dependent anomalous scatter factors for a single element.

    INPUTS:
    elem - short element name (str)
           e.g. for Hydrogen, elem='H'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    
    OUTPUTS:
    f1, f2 - arrays of scatter factors as function of energy
    """
    try:
        len(energy)
        energy = np.array(energy)
    except:
        pass
        
    n = 10  # number of characters per column in data file
    with open(get_filename(elem)) as f:
        data = np.array([[line[i:i+n] for i in range(0, len(line)-1, n)] for line in f.readlines()[2:]]).T

    # Get table of keV, f1, f2
    keV_0 = 1000 * np.array([to_float(x) for x in data[0]])
    f1_0 = np.array([to_float(x) for x in data[2]])
    f2_0 = np.array([to_float(x) for x in data[5]])
    
    # Linear interp to energy vals
    f1 = np.interp(energy, keV_0, f1_0)
    f2 = np.interp(energy, keV_0, f2_0)
    
    return f1, f2


def get_delta_beta(elem, energy, density):
    """
    Get energy-dependent phase (delta) and absorption (beta) parameters.

    INPUTS:
    elem - short element name (str)
           e.g. for Hydrogen, elem='H'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    density - element density in g/cm^3 (float)
    
    OUTPUTS:
    delta, beta - arrays of scatter factors as function of energy
    """
    # Get some scaling values
    A = get_amass(elem) # atomic mass
    n_a = (10**6)*density*N_A/A               # number density
    alpha = n_a*r_e/(2*pi)            # scale factor
    wavelen = get_wavelen(energy)    # wavelengths [m]

    # Get the scatter factors
    f1, f2 = get_f1_f2(elem, energy)
    delta = alpha * wavelen**2 * f1
    beta = alpha * wavelen**2 * f2

    return delta, beta




# ## HANDLING MIXTURES - USE STOICHIOMETRIC WEIGHTING
# ## Weighted average of atomic masses
# ## Implemented following Jacobsen ch 3, eq'n 3.101-103


def parse_matcomp(name):
    """
    Converts string of material composition to list of material names 
    and a list of their weightings.
    
    INPUTS:
    name - matcomp string, e.g. for water 'H(88.8)O(11.2)'

    OUTPUTS:
    matnames - list of individual element names, e.g. ['H','O']
    weights - list of corresponding weights, e.g. [88.8, 11.2]
    """
    
    matnames = []
    weights = []

    ii = 0
    sub = name
    lp = sub.find('(')
    rp = sub.find(')')
    while lp != -1:
        matnames.append(sub[:lp])
        weights.append(float(sub[(lp+1):rp]))
        ii = rp+1
        sub = sub[ii:]
        lp = sub.find('(')
        rp = sub.find(')')
        
    # Normalize weights
    weights = np.array(weights)
    weights = weights/np.sum(weights)
    
    return matnames, weights 


def get_f1_f2_mix(matcomp, energy):
    """
    Get energy-dependent anomalous scatter factors for a mixture
    of elements with given weighting factors.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    
    OUTPUTS:
    f1, f2 - arrays of scatter factors as function of energy
    """
    matnames, weights = parse_matcomp(matcomp)
    N_elems = len(matnames)
    N_energy = np.array(energy).size
    
    if N_elems == 1:  # For one element, no need to mix
        return get_f1_f2(matnames[0], energy)
    
    # Else get f1, f2 for each element
    # Squeeze in case N_energy == 1
    f1_elems_weighted = np.zeros([N_elems, N_energy]).squeeze()  
    f2_elems_weighted = np.zeros([N_elems, N_energy]).squeeze()
    for i in range(N_elems):
        elem = matnames[i]
        s_i = weights[i]
        f1, f2 = get_f1_f2(elem, energy)
        f1_elems_weighted[i] = s_i * f1
        f2_elems_weighted[i] = s_i * f2

    # Compute weighted averages as func of energy
    f1_avg = np.sum(f1_elems_weighted, axis=0)
    f2_avg = np.sum(f2_elems_weighted, axis=0)

    return f1_avg, f2_avg


def get_amass_mix(matcomp):
    """
    Finds the average atomic mass of a mixture.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    
    OUTPUTS:
    amass - double specifiying avg atomic mass in grams per mole
    """
    matnames, weights = parse_matcomp(matcomp)
    N_elems = len(matnames)

    amass_elems_weighted = np.zeros(N_elems)
    for i in range(N_elems):
        elem = matnames[i]
        s_i = weights[i]
        amass = get_amass(elem)
        amass_elems_weighted[i] = s_i * amass

    # compute weighted average
    amass = np.sum(amass_elems_weighted)

    return amass



def get_delta_beta_mix(matcomp, energy, density):
    """
    Get energy-dependent phase (delta) and absorption (beta) parameters
    for a mixture of elements with given weighting factors.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    density - element density in g/cm^3 (float)
    
    OUTPUTS:
    delta, beta - arrays of scatter factors as function of energy
    """
    matnames, weights = parse_matcomp(matcomp)
    N_elems = len(matnames)
    N_energy = np.array(energy).size

    # check for mixture or single
    if N_elems == 1: # single element
        elem = matnames[0]
        A = get_amass(elem) # atomic mass
        f1, f2 = get_f1_f2(elem, energy)
    else:
        A = get_amass_mix(matcomp)
        f1, f2 = get_f1_f2_mix(matcomp, energy)

    # some scale factors
    n_a = (10**6)*density*N_A/A      # number density 
    alpha = n_a*r_e/(2*pi)   # scale factor 
    wavelen = get_wavelen(energy)    # wavelengths [m]

    # compute the scatter factors
    delta = alpha * wavelen**2 * f1
    beta = alpha * wavelen**2 * f2

    return delta, beta


def mix_matcomp(matcomp1, density1, m1, matcomp2, density2, m2):
    """
    Mix two materials into one solution.
    Assumptions: mi = sum[mi], Vi = sum[Vi]
    x1, x2 = mass fractions of total solution
    """
    matnames1, weights1 = parse_matcomp(matcomp1)
    matnames2, weights2 = parse_matcomp(matcomp2)
    density = (m1 + m2)/((m1/density1) + (m2/density2))  # see assumptions
    # density = 1/(x1/density1 + x2/density2)
    
    weights1_scaled = weights1*density1/density
    weights2_scaled = weights2*density2/density

    matnames = list(set(matnames1 + matnames2)) # unique materials
    weights = np.zeros(len(matnames))
    for i, mat in enumerate(matnames):
        if mat in matnames1:
            weights[i] += weights1_scaled[matnames1.index(mat)]
        if mat in matnames2:
            weights[i] += weights2_scaled[matnames2.index(mat)]

    weights = weights/np.sum(weights)  # normalize weights

    # make xcompy-style string
    matcomp = ''.join([f'{mat}({weight:.6f})' for mat,weight in zip(matnames, weights)])
    return matcomp, density


def get_dbm_mix(matcomp, energy, density):
    """
    Get energy-dependent delta, beta, and mu (linear atten. coefficient)
    for a mixture of elements with given weighting factors.
    Note mu is just dependent on beta and the wavenumber for the energies.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    density - element density in g/cm^3 (float)
    
    OUTPUTS:
    delta, beta, mu - arrays of scatter factors as function of energy
    """
    delta, beta = get_delta_beta_mix(matcomp, energy, density)
    wavenum = get_wavenum(energy)
    mu = 2 * wavenum * beta
    return delta, beta, mu


class Material:
    def __init__(self, name, matcomp, density):
        self.name = name
        self.matcomp = matcomp
        self.density = density

        # pre-calculate delta, beta at a representative range of energies to avoid file reloading
        self.energy_range = jnp.linspace(1, 150, 150)
        self.delta_range, self.beta_range = get_delta_beta_mix(matcomp, self.energy_range, density)
        
    def delta_beta(self, energy):
        """
        Returns linearly interpolated delta and beta at the given energy.
        """
        delta = jnp.interp(energy, self.energy_range, self.delta_range)
        beta = jnp.interp(energy, self.energy_range, self.beta_range)
        return delta, beta


        




    