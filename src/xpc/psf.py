import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.signal import convolve2d

PI = jnp.pi

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


def apply_psf(img, FOV, dx, fwhm=None, kernel_width=1, psf='lorentzian'):
    """ Apply a PSF to an image."""

    # Check special condition
    if fwhm is None:
        return img

    # Check if PSF format is supported
    psf = psf.lower()
    assert psf in ('lorentzian', 'gaussian')

    # Compute the reduced FOV for kernel grid, for efficiency
    small_FOV = kernel_width * FOV   # reduce kernel size to improve convolution time
    x = jnp.arange(-small_FOV, small_FOV, dx) + dx

    # Generate the kernel (normalized by default)
    if psf == 'lorentzian':
        kernel = lorentzian2D(x, x, fwhm)
    elif psf == 'gaussian':
        kernel = gaussian2D(x, x, fwhm)
        
    img_pad = jnp.pad(img, kernel.shape, constant_values=img[0,0])    # pad img to account for fillvalue = 0. Corner [0,0] pixel might be bad idea?
    img_nonideal_pad = convolve2d(img_pad, kernel, mode='same')
    img_nonideal = img_nonideal_pad[kernel.shape[0]:-kernel.shape[0], kernel.shape[1]:-kernel.shape[1]]
        
    return img_nonideal




