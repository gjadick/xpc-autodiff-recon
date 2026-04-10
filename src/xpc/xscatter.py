#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 6 12:05:10 2023
Last updated on Thu Aug 5 12:37:00 2024

@author: gjadick

X-ray optics helper functions and other things (e.g. material compositions).

    
"""

import numpy as np
import jax.numpy as jnp
import xraydb


### CONSTANTS
r_e = 2.8179403262e-15        # classical electron radius, m
N_A = 6.02214076e+23          # Avogadro's number, num/mol
PI = np.pi
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
    return 2*PI / get_wavelen(energy)


def get_energy(wavelen):
    """wavelen in m --> returns energy in keV"""
    try:
        len(wavelen)
        wavelen = np.array(wavelen)
    except:
        pass  
    return 1e-3*h*c / (wavelen*J_eV)


def gaussian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Gaussian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Gaussian (units must match x,y)
    normalize: bool (default True)
        If True, normalize the kernel to sum to 1
    """
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    X, Y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel

    
def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    normalize: bool (default True)
        If True, normalize the kernel to sum to 1
    """
    gamma = fwhm/2
    X, Y = jnp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def convolve2d_diy(img, kernel):
    """Convolve two 2D arrays with scipy-like mode='same'."""
    H, W = img.shape
    KH, KW = kernel.shape
    F = jnp.fft.rfft2(img, s=(H+KH-1, W+KW-1))
    G = jnp.fft.rfft2(kernel, s=(H+KH-1, W+KW-1))
    out = jnp.fft.irfft2(F * G, s=(H+KH-1, W+KW-1))
    oh, ow = (KH-1)//2, (KW-1)//2   # center crop (mode='same')
    return out[oh:oh+H, ow:ow+W]

    
def apply_psf(img, dx, fwhm=None, kernel_width=6.0, psf='lorentzian'):
    if fwhm is None:
        return img

    psf = psf.lower()
    assert psf in ('lorentzian', 'gaussian')

    half_width = kernel_width * fwhm        
    x = jnp.arange(-half_width, half_width + dx, dx)

    if psf == 'lorentzian':
        kernel = lorentzian2D(x, x, fwhm)
    else:
        kernel = gaussian2D(x, x, fwhm)

    img_pad = jnp.pad(img, kernel.shape, constant_values=img[0,0])    # pad img to account for fillvalue = 0. Corner [0,0] pixel temp
    img_nonideal_pad = convolve2d_diy(img_pad, kernel)
    img_nonideal = img_nonideal_pad[kernel.shape[0]:-kernel.shape[0], kernel.shape[1]:-kernel.shape[1]]
        
    return img_nonideal
    

class Material:

    def __init__(self, name, matcomp, density, k_blur=1.0, thresh=0.5, E_min=1.0, E_max=100.0):
        """     
        name: material identifier (str)
        matcomp: NIST-style material composition and weights (str)
                  e.g. for water 'H(88.8)O(11.2)'
        density: element density in g/cm^3 (float)
        """
        # Material parameters
        self.name = name
        self.matcomp = matcomp
        self.density = float(density)

        # Phantom upsampling parameters -- tune to produce realistic morphology
        self.k_blur = k_blur
        self.thresh = thresh

        # Convert NIST weight% matcomp to xraydb style:
        self.formula_xraydb = self._wtpct_to_xraydb_formula(self.matcomp)

        # Pre-calc delta/beta on an energy grid (keV) to avoid repeated xraydb calls
        self.energy_range = np.arange(E_min, E_max, 1.0)  # keV
        d_np, b_np = self._delta_beta_xraydb(np.asarray(self.energy_range))
        self.delta_range = np.asarray(d_np)
        self.beta_range = np.asarray(b_np)

    @staticmethod
    def _parse_matcomp_wtpct(matcomp: str):
        """
        Parse strings like 'H(10.2)C(14.3)...' into (elements, weights),
        where weight percentages are normalized mass fractions summing to 1.
        """
        elems, wts = [], []

        sub = matcomp.strip()
        lp = sub.find("(")
        rp = sub.find(")")

        while lp != -1:
            elems.append(sub[:lp])
            wts.append(float(sub[lp + 1 : rp]))
            sub = sub[rp + 1 :].strip()
            lp = sub.find("(")
            rp = sub.find(")")

        wts = np.asarray(wts, dtype=float)
        wts = wts / wts.sum()  # normalize to mass fractions
        return elems, wts

    @staticmethod
    def _wtpct_to_xraydb_formula(matcomp_wtpct, scale=100.0, fmt='.8g'):
        """
        Convert NIST-style matcomp (weight percent) into xraydb-style chemical formula (atomic ratios)
        """
        elems, w = Material._parse_matcomp_wtpct(matcomp_wtpct)
        A = np.array([xraydb.atomic_mass(el) for el in elems], dtype=float)  # g/mol
        mol = w / A
        mol_frac = mol / mol.sum()
        coeff = mol_frac * float(scale)   # scale cancels out, helps with very tiny fractions
        parts = [f'{el}{format(ci, fmt)}' for el, ci in zip(elems, coeff)]
        return ''.join(parts)

    def _delta_beta_xraydb(self, energy_keV):
        """
        energy_keV: numpy array of energies in keV
        returns: (delta, beta) numpy arrays
        """
        energy_eV = 1e3 * np.asarray(energy_keV, dtype=float)
        delta, beta, _atlen = xraydb.xray_delta_beta(self.formula_xraydb, self.density, energy_eV)
        return np.asarray(delta, dtype=float), np.asarray(beta, dtype=float)

    def delta_beta(self, energy):
        """
        Return (delta, beta) at energy [keV] via interpolation of precomputed grid.
        Works for scalar or array-like energy.
        """
        delta = np.interp(energy, self.energy_range, self.delta_range)
        beta  = np.interp(energy, self.energy_range, self.beta_range)
        return delta, beta



## NIST material definitions -- https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html
nist_mc = {
    'A-150 Tissue-Equivalent Plastic': 'H(0.101330)C(0.775498)N(0.035057)O(0.052315)F(0.017423)Ca(0.018377)',
    'Adipose Tissue (ICRU-44)': 'H(0.114000)C(0.598000)N(0.007000)O(0.278000)Na(0.001000)S(0.001000)Cl(0.001000)',
    'Air, Dry (near sea level)': 'C(0.000124)N(0.755268)O(0.231781)Ar(0.012827)',
    'Alanine': 'H(0.079192)C(0.404437)N(0.157213)O(0.359157)',
    'B-100 Bone-Equivalent Plastic': 'H(0.065473)C(0.536942)N(0.021500)O(0.032084)F(0.167415)Ca(0.176585)',
    'Bakelite': 'H(0.057444)C(0.774589)O(0.167968)',
    'Blood, Whole (ICRU-44)': 'H(0.102000)C(0.110000)N(0.033000)O(0.745000)Na(0.001000)P(0.001000)S(0.002000)Cl(0.003000)K(0.002000)Fe(0.001000)',
    'Bone, Cortical (ICRU-44)': 'H(0.034000)C(0.155000)N(0.042000)O(0.435000)Na(0.001000)Mg(0.002000)P(0.103000)S(0.003000)Ca(0.225000)',
    'Brain, Grey/White Matter (ICRU-44)': 'H(0.107000)C(0.145000)N(0.022000)O(0.712000)Na(0.002000)P(0.004000)S(0.002000)Cl(0.003000)K(0.003000)',
    'Breast Tissue (ICRU-44)': 'H(0.106000)C(0.332000)N(0.030000)O(0.527000)Na(0.001000)P(0.001000)S(0.002000)Cl(0.001000)',
    'C-552 Air-equivalent Plastic': 'H(0.024681)C(0.501610)O(0.004527)F(0.465209)Si(0.003973)',
    'Cadmium Telluride': 'Cd(0.468358)Te(0.531642)',
    'Calcium Fluoride': 'F(0.486672)Ca(0.513328)',
    'Calcium Sulfate': 'O(0.470081)S(0.235534)Ca(0.294385)',
    '15 mmol L-1 Ceric Ammonium Sulfate Solution': 'H(0.107694)N(0.000816)O(0.875172)S(0.014279)Ce(0.002040)',
    'Cesium Iodide': 'I(0.488451)Cs(0.511549)',
    'Concrete, Ordinary': 'H(0.022100)C(0.002484)O(0.574930)Na(0.015208)Mg(0.001266)Al(0.019953)Si(0.304627)K(0.010045)Ca(0.042951)Fe(0.006435)',
    'Concrete, Barite (TYPE BA)': 'H(0.003585)O(0.311622)Mg(0.001195)Al(0.004183)Si(0.010457)S(0.107858)Ca(0.050194)Fe(0.047505)Ba(0.463400)',
    'Eye Lens (ICRU-44)': 'H(0.096000)C(0.195000)N(0.057000)O(0.646000)Na(0.001000)P(0.001000)S(0.003000)Cl(0.001000)',
    'Ferrous Sulfate Standard Fricke': 'H(0.108376)O(0.878959)Na(0.000022)S(0.012553)Cl(0.000035)Fe(0.000055)',
    'Gadolinium Oxysulfide': 'O(0.084527)S(0.084704)Gd(0.830769)',
    'Gafchromic Sensor': 'H(0.089700)C(0.605800)N(0.112200)O(0.192300)',
    'Gallium Arsenide': 'Ga(0.482030)As(0.517970)',
    'Glass, Borosilicate (Pyrex)': 'B(0.040066)O(0.539559)Na(0.028191)Al(0.011644)Si(0.377220)K(0.003321)',
    'Glass, Lead': 'O(0.156453)Si(0.080866)Ti(0.008092)As(0.002651)Pb(0.751938)',
    'Lithium Fluride': 'Li(0.267585)F(0.732415)',
    'Lithium Tetraborate': 'Li(0.082081)B(0.255715)O(0.662204)',
    'Lung Tissue (ICRU-44)': 'H(0.103000)C(0.105000)N(0.031000)O(0.749000)Na(0.002000)P(0.002000)S(0.003000)Cl(0.003000)K(0.002000)',
    'Magnesium Tetroborate': 'B(0.240870)O(0.623762)Mg(0.135367)',
    'Mercuric Iodide': 'I(0.558560)Hg(0.441440)',
    'Muscle, Skeletal (ICRU-44)': 'H(0.102000)C(0.143000)N(0.034000)O(0.710000)Na(0.001000)P(0.002000)S(0.003000)Cl(0.001000)K(0.004000)',
    'Ovary (ICRU-44)': 'H(0.105000)C(0.093000)N(0.024000)O(0.768000)Na(0.002000)P(0.002000)S(0.002000)Cl(0.002000)K(0.002000)',
    'Photographic Emulsion (Kodak Type AA)': 'H(0.030500)C(0.210700)N(0.072100)O(0.163200)Br(0.222800)Ag(0.300700)',
    'Photographic Emulsion (Standard Nuclear)': 'H(0.014100)C(0.072261)N(0.019320)O(0.066101)S(0.001890)Br(0.349104)Ag(0.474105)I(0.003120)',
    'Plastic Scintillator, Vinyltoluene': 'H(0.085000)C(0.915000)',
    'Polyethylene': 'H(0.143716)C(0.856284)',
    'Polyethylene Terephthalate, (Mylar)': 'H(0.041960)C(0.625016)O(0.333024)',
    'Polymethyl Methacrylate': 'H(0.080541)C(0.599846)O(0.319613)',
    'Polystyrene': 'H(0.077421)C(0.922579)',
    'Polytetrafluoroethylene, (Teflon)': 'C(0.240183)F(0.759818)',
    'Polyvinyl Chloride': 'H(0.048382)C(0.384361)Cl(0.567257)',
    'Radiochromic Dye Film, Nylon Base': 'H(0.101996)C(0.654396)N(0.098915)O(0.144693)',
    'Testis (ICRU-44)': 'H(0.106000)C(0.099000)N(0.020000)O(0.766000)Na(0.002000)P(0.001000)S(0.002000)Cl(0.002000)K(0.002000)',
    'Tissue, Soft (ICRU-44)': 'H(0.102000)C(0.143000)N(0.034000)O(0.708000)Na(0.002000)P(0.003000)S(0.003000)Cl(0.002000)K(0.003000)',
    'Tissue, Soft (ICRU Four-Component)': 'H(0.101174)C(0.111000)N(0.026000)O(0.761826)',
    'Tissue-Equivalent Gas, Methane Based': 'H(0.101873)C(0.456177)N(0.035172)O(0.406778)',
    'Tissue-Equivalent Gas, Propane Based': 'H(0.102676)C(0.568937)N(0.035022)O(0.293365)',
    'Water, Liquid': 'H(0.111898)O(0.888102)',
}


## Breast microcalcification material compositions, two types:
calc_mc = {
    'Calcium Oxalate (Dihydrate)': 'Ca(24.43)C(14.64)O(58.49)H(2.44)',  # type 1, density ~ 2.0 g/cm3
    'Hydroxyapatite': 'Ca(39.89)P(18.50)O(41.41)H(0.20)',   # type 2, density ~ 3.1 g/cm3

}

## VICTRE phantom materials, mapping given voxelId to NIST chemical compositions.
##   Based on pipeline defaults -- https://github.com/DIDSR/VICTRE_PIPELINE/blob/main/Victre/Constants.py
##   and densities used in -- https://github.com/DIDSR/VICTRE_MCGPU/blob/master/MC-GPU_v1.5b_sample_mammo_and_DBT_simulation_InputDensity.in
victre_matdict = {  
    0: Material(
        'air', 
        density = 0.0012,
        matcomp = nist_mc['Air, Dry (near sea level)'],
    ),
    1: Material(
        'adipose', 
        density = 0.92,
        matcomp = nist_mc['Adipose Tissue (ICRU-44)'],
        k_blur = 1.5, 
    ),
    2: Material(
        'skin', 
        density = 1.09,
        matcomp = nist_mc['Tissue, Soft (ICRU-44)'],
        k_blur = 0.6
    ),
    29: Material(
        'glandular',
        density = 1.035,
        matcomp = nist_mc['Breast Tissue (ICRU-44)'],
    ),
    33: Material(
        'nipple',
        density = 1.09,
        matcomp = nist_mc['Tissue, Soft (ICRU-44)'],
    ),
    40: Material(
        'muscle',
        density = 1.05,
        matcomp = nist_mc['Muscle, Skeletal (ICRU-44)'],
    ),
    50: Material(
        'paddle',
        density = 1.06,
        matcomp = nist_mc['Polystyrene'],
    ),
    88: Material(
        'ligament', 
        density = 1.12,
        matcomp = nist_mc['Tissue, Soft (ICRU-44)'],
        thresh = 0.2
    ),
    95: Material(
        'TDLU',
        density = 1.05,
        matcomp = nist_mc['Muscle, Skeletal (ICRU-44)'],
    ),
    125: Material(
        'duct', 
        density = 1.05,
        matcomp = nist_mc['Muscle, Skeletal (ICRU-44)'],
        thresh = 0.5
    ),
    150: Material(
        'artery',
        density = 1.0,
        matcomp = nist_mc['Blood, Whole (ICRU-44)'],
    ),
    200: Material(
        'spiculated',
        density = 1.06,
        matcomp = nist_mc['Breast Tissue (ICRU-44)'],
    ),
    225: Material(
        'vein',
        density = 1.0,
        matcomp = nist_mc['Blood, Whole (ICRU-44)'],
    ),
    250: Material(
        'clustercalc', 
        density = 2.0,
        matcomp = calc_mc['Calcium Oxalate (Dihydrate)'], 
        k_blur = 0.5, 
        thresh = 0.5
    ),
}

