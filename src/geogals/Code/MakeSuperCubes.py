import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage, interpolate
from skimage.registration import phase_cross_correlation
from astropy.wcs import WCS
import warnings
import os 
from photutils.centroids import centroid_com
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.psf import IntegratedGaussianPRF
import matplotlib.patches as patches

galname = 'UGC05101'
centerpix = [73,66]

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_datacubes():
    """Load available datacubes and return them as a list of data arrays and headers"""
    path = '.'
    cubes = []
    headers = []
    names = []
    
    # Mapping of filename patterns to short names
    file_patterns = {
        'BH1L': 'BH1',
        'BH3L': 'BH3', 
        'RH1L': 'RH1',
        'RH2L': 'RH2',
        'BLL': 'BLL',
        'BML': 'BML'
    }
    
    # Load all files
    dirlist = os.listdir(path)
    for file in dirlist:
        if '.fits' in file:
            for pattern, name in file_patterns.items():
                if pattern in file:
                    try:
                        fits_file = fits.open(file)
                        cubes.append(fits_file[0].data)
                        headers.append(fits_file[0].header)
                        names.append(name)
                        print(f"Successfully loaded {file} as {name}")
                        break
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
    
    if len(cubes) == 0:
        raise ValueError("No FITS files found with the expected patterns (BH1L, BH3L, RH1L, RH2L)")
    
    print(f"Loaded {len(cubes)} cubes: {names}")
    return cubes, headers, names

def analyze_galaxy(supercube, new_wave, white_image, output_path, center=[49, 49]):
    """
    Simplified galaxy analysis with predefined center
    """
    # 1. Use provided center coordinates
    cx, cy = center
    print(f"Using provided center at x={cx:.1f}, y={cy:.1f}")

    # 2. Calculate radial profile
    yy, xx = np.indices(white_image.shape)
    radii = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    # Bin the radii and sum fluxes in each bin
    max_radius = min(white_image.shape)//2
    radial_bins = np.linspace(0, max_radius, 50)
    profile = []
    area = []
    
    for i in range(len(radial_bins)-1):
        mask = (radii >= radial_bins[i]) & (radii < radial_bins[i+1]) & ~np.isnan(white_image)
        profile.append(np.nansum(white_image[mask]))
        area.append(np.sum(mask))  # Actual pixel count (accounts for NaNs)
    
    profile = np.array(profile)
    area = np.array(area)
    mean_radii = (radial_bins[:-1] + radial_bins[1:])/2

    # 3. Calculate cumulative flux profile
    with np.errstate(invalid='ignore'):
        cumulative_flux = np.cumsum(profile)
        total_flux = np.nansum(profile)
        if total_flux > 0:
            norm_cumulative = cumulative_flux / total_flux
        else:
            norm_cumulative = np.zeros_like(cumulative_flux)
    
    # 4. Find half-light radius (where cumulative flux reaches 50%)
    if len(norm_cumulative) > 0:
        half_light_idx = np.argmin(np.abs(norm_cumulative - 0.5))
        r_half = mean_radii[half_light_idx]
    else:
        r_half = max_radius/4  # Fallback value
    
    print(f"Half-light radius: {r_half:.1f} pixels")

    # 5. Create diagnostic plots
    plt.figure(figsize=(15, 5))
    
    # White light image with aperture
    plt.subplot(131)
    plt.imshow(white_image, origin='lower', cmap='viridis',
              vmax=np.nanpercentile(white_image, 99))
    plt.colorbar(label='Flux')
    plt.scatter(cx, cy, color='red', marker='+', s=100, lw=2)
    plt.gca().add_patch(plt.Circle((cx, cy), r_half, color='red', 
                       fill=False, lw=2, linestyle='--'))
    plt.gca().add_patch(plt.Circle((cx, cy), r_half/2, color='lime', 
                       fill=False, lw=2, linestyle='--'))
    plt.title(f'Half-light radius = {r_half:.1f} pix')

    # Radial profile
    plt.subplot(132)
    plt.plot(mean_radii, profile/area, 'b-')  # Surface brightness profile
    plt.axvline(r_half, color='r', linestyle='--')
    plt.xlabel('Radius [pix]')
    plt.ylabel('Mean Surface Brightness')
    plt.title('Radial Profile')
    plt.grid(True, alpha=0.3)

    # Spectrum within half-light radius
    aperture_mask = radii <= r_half/2
    half_light_spectrum = np.nansum(supercube * aperture_mask[np.newaxis,:,:], axis=(1, 2))
    
    plt.subplot(133)
    plt.plot(new_wave, half_light_spectrum, 'k-', lw=1)
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Flux')
    plt.title('Spectrum Within Half the Effective Radius')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'galaxy_analysis.png'), 
               bbox_inches='tight', dpi=150)
    plt.show()
    
    return cx, cy, r_half

def get_wavelength_info(headers):
    """Extract wavelength information from headers"""
    wave_info = []
    for hdr in headers:
        try:
            naxis3 = hdr['NAXIS3']
            crval3 = hdr['CRVAL3']  # Starting wavelength (Å)
            cdelt3 = hdr['CD3_3']  # Wavelength step (Å)
            crpix3 = hdr['CRPIX3']  # Reference pixel
            
            # Create wavelength array
            wave = crval3 + (np.arange(naxis3) - crpix3) * cdelt3
            wave_info.append({
                'wave': wave,
                'min': wave[0],
                'max': wave[-1],
                'step': cdelt3,
                'n': naxis3
            })
        except KeyError as e:
            print(f"Warning: Missing wavelength information in header: {e}")
            # Add placeholder info
            wave_info.append({
                'wave': np.arange(naxis3),
                'min': 0,
                'max': naxis3-1,
                'step': 1,
                'n': naxis3
            })
    
    return wave_info

def create_white_images(cubes):
    """Create white light images by summing along wavelength axis"""
    white_images = []
    for cube in cubes:
        # Sum along wavelength axis, ignoring NaN values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            white = np.nansum(cube, axis=0)
        white_images.append(white)
    
    return white_images

def get_transformations(names):
    """
    Prompt user to specify transformations for each cube
    Returns a dictionary mapping cube names to transformation parameters
    """
    transformations = {}
    
    print("\n=== Specify Cube Transformations ===")
    print("Enter transformations for each cube:")
    print("Rotation: angle in degrees (0 for no rotation)")
    print("Flip: 'x' for horizontal flip, 'y' for vertical flip, 'none' for no flip")
    print("Positive rotation angles = counterclockwise, Negative = clockwise")
    print("Common rotations: 90, 180, 270, -90\n")
    
    for name in names:
        print(f"\n--- Transformations for {name} ---")
        
        # Get rotation angle
        while True:
            try:
                angle = float(input(f"Rotation angle for {name} (degrees): "))
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Get flip direction
        while True:
            flip = input(f"Flip direction for {name} (x/y/none): ").lower().strip()
            if flip in ['x', 'y', 'none', '']:
                if flip == '':
                    flip = 'none'
                break
            else:
                print("Please enter 'x', 'y', or 'none'")
        
        transformations[name] = {
            'rotation_angle': angle,
            'flip_direction': flip
        }
        
        # Print summary
        transform_desc = []
        if angle != 0:
            transform_desc.append(f"rotate {angle}°")
        if flip != 'none':
            transform_desc.append(f"flip {flip}")
        if not transform_desc:
            transform_desc.append("no transformation")
        
        print(f"{name}: {', '.join(transform_desc)}")
    
    return transformations

def transform_image(image, rotation_angle, flip_direction):
    """Apply rotation and flip transformations to an image"""
    # Replace NaN values with 0 for transformation
    image_no_nan = np.nan_to_num(image, nan=0.0)
    
    # Apply rotation using interpolation for arbitrary angles
    if rotation_angle != 0:
        # Rotate the image
        image_no_nan = ndimage.rotate(image_no_nan, rotation_angle, reshape=True, 
                                     mode='constant', cval=0.0, order=1)
    
    # Apply flip
    if flip_direction == 'x':
        image_no_nan = np.fliplr(image_no_nan)
    elif flip_direction == 'y':
        image_no_nan = np.flipud(image_no_nan)
    
    return image_no_nan

def transform_cube(cube, rotation_angle, flip_direction):
    """Apply rotation and flip transformations to a datacube (spatial axes only)"""
    # Replace NaN values with 0 for transformation
    cube_no_nan = np.nan_to_num(cube, nan=0.0)
    
    # Apply rotation using interpolation for arbitrary angles
    if rotation_angle != 0:
        # Rotate the first slice to determine the new shape
        first_slice_rotated = ndimage.rotate(cube_no_nan[0], rotation_angle, reshape=True, 
                                           mode='constant', cval=0.0, order=1)
        
        # Create a new array with the correct shape
        rotated_cube = np.zeros((cube_no_nan.shape[0], first_slice_rotated.shape[0], 
                               first_slice_rotated.shape[1]))
        
        # Rotate each wavelength slice
        for i in range(cube_no_nan.shape[0]):
            rotated_cube[i] = ndimage.rotate(cube_no_nan[i], rotation_angle, reshape=True, 
                                           mode='constant', cval=0.0, order=1)
        cube_no_nan = rotated_cube
    
    # Apply flip
    if flip_direction == 'x':
        cube_no_nan = np.flip(cube_no_nan, axis=2)  # Flip along x-axis (axis 2)
    elif flip_direction == 'y':
        cube_no_nan = np.flip(cube_no_nan, axis=1)  # Flip along y-axis (axis 1)
    
    return cube_no_nan

def align_images(white_images, names, transformations):
    """Align images with user-specified transformations"""
    if len(white_images) <= 1:
        print("Only one image found, no alignment needed")
        return white_images, [np.array([0, 0])], white_images[0].shape
    
    # Apply transformations to white images
    transformed_white_images = []
    for img, name in zip(white_images, names):
        params = transformations[name]
        transformed_img = transform_image(img, params['rotation_angle'], params['flip_direction'])
        transformed_white_images.append(transformed_img)
    
    # Find maximum dimensions across all transformed images
    max_shape = np.max([img.shape for img in transformed_white_images], axis=0)
    
    # Pad all images to max_shape
    padded_images = []
    for img in transformed_white_images:
        pad_y = max_shape[0] - img.shape[0]
        pad_x = max_shape[1] - img.shape[1]
        padded = np.pad(img, ((0, pad_y), (0, pad_x)), mode='constant')
        padded_images.append(padded)
    
    # Use the first image as reference
    reference = padded_images[0]
    aligned_images = [reference.copy()]
    shifts = [np.array([0, 0])]
    
    fig, axes = plt.subplots(1, len(white_images), figsize=(4*len(white_images), 4))
    if len(white_images) == 1:
        axes = [axes]
    
    # Plot reference image
    axes[0].imshow(reference, origin='lower', cmap='viridis')
    axes[0].set_title(f'{names[0]} (Reference)')
    
    for i, (img, name) in enumerate(zip(padded_images[1:], names[1:]), 1):
        # Create masks to exclude NaN values
        reference_mask = ~np.isnan(reference)
        moving_mask = ~np.isnan(img)
        
        # Find shift between current image and reference
        shift, error, diffphase = phase_cross_correlation(
            reference, img, 
            reference_mask=reference_mask,
            moving_mask=moving_mask,
            upsample_factor=10
        )
        
        shift = np.array(shift)
        aligned_img = ndimage.shift(img, shift)
        aligned_images.append(aligned_img)
        shifts.append(shift)
        
        # Plot results
        axes[i].imshow(aligned_img, origin='lower', cmap='viridis')
        axes[i].set_title(f'{name} - Shift: {shift}')
    
    plt.tight_layout()
    plt.show()
    
    return aligned_images, shifts, max_shape

def apply_spatial_shifts(cubes, shifts, max_shape, names, transformations):
    """Apply spatial shifts with user-specified transformations"""
    shifted_cubes = []
    for cube, shift, name in zip(cubes, shifts, names):
        # Apply transformations to cube
        params = transformations[name]
        transformed_cube = transform_cube(cube, params['rotation_angle'], params['flip_direction'])
        
        # Pad the transformed cube to max_shape
        pad_y = max_shape[0] - transformed_cube.shape[1]
        pad_x = max_shape[1] - transformed_cube.shape[2]
        padded_cube = np.pad(transformed_cube, ((0, 0), (0, pad_y), (0, pad_x)), mode='constant')
        
        if np.allclose(shift, 0):
            shifted_cubes.append(padded_cube)
            continue
            
        # Shift each wavelength slice
        shifted_cube = np.zeros_like(padded_cube)
        for j in range(padded_cube.shape[0]):
            shifted_cube[j] = ndimage.shift(padded_cube[j], shift)
        
        shifted_cubes.append(shifted_cube)
    
    return shifted_cubes

def resample_spectra(cubes, wave_info):
    """Resample spectra to highest resolution and track wavelength ranges"""
    if len(cubes) <= 1:
        print("Only one cube found, no resampling needed")
        return cubes, wave_info[0]['wave'], [(0, len(wave_info[0]['wave'])-1)]
    
    # Find the cube with highest spectral resolution (smallest step)
    steps = [info['step'] for info in wave_info]
    highest_res_idx = np.argmin(np.abs(steps))
    highest_res_step = wave_info[highest_res_idx]['step']
    
    print(f"Highest resolution cube has step {highest_res_step:.2f} Å")
    
    # Get the full wavelength range from all cubes
    all_min = min(info['min'] for info in wave_info)
    all_max = max(info['max'] for info in wave_info)
    
    # Create new wavelength grid with highest resolution
    new_wave = np.arange(all_min, all_max + highest_res_step, highest_res_step)
    
    # Prepare for resampling and track wavelength ranges
    resampled_cubes = []
    wave_ranges = []
    
    for i, (cube, info) in enumerate(zip(cubes, wave_info)):
        # Find where this cube's wavelengths fit in the new grid
        start_idx = np.argmin(np.abs(new_wave - info['min']))
        end_idx = np.argmin(np.abs(new_wave - info['max']))
        wave_ranges.append((start_idx, end_idx))
        
        if i == highest_res_idx:
            # No need to resample the highest resolution cube
            resampled_cubes.append(cube)
            continue
            
        print(f"Resampling cube {i} from {info['step']:.2f} Å to {highest_res_step:.2f} Å")
        
        # Create empty array for resampled cube
        resampled = np.zeros((end_idx - start_idx + 1, cube.shape[1], cube.shape[2]))
        
        # Resample each spaxel
        for y in range(cube.shape[1]):
            for x in range(cube.shape[2]):
                interp_func = interpolate.interp1d(
                    info['wave'], cube[:, y, x],
                    kind='linear', bounds_error=False, fill_value=0.0
                )
                resampled[:, y, x] = interp_func(new_wave[start_idx:end_idx+1])
        
        resampled_cubes.append(resampled)
    
    return resampled_cubes, new_wave, wave_ranges

def create_supercube(resampled_cubes, new_wave, wave_ranges):
    """Combine resampled cubes into a supercube using wavelength ranges"""
    if len(resampled_cubes) == 1:
        print("Only one cube found, returning it as supercube")
        return resampled_cubes[0]
    
    # Find maximum spatial dimensions
    max_y = max(cube.shape[1] for cube in resampled_cubes)
    max_x = max(cube.shape[2] for cube in resampled_cubes)
    
    # Create empty supercube
    supercube = np.zeros((len(new_wave), max_y, max_x))
    
    # Combine data using the wavelength ranges
    for cube, (start_idx, end_idx) in zip(resampled_cubes, wave_ranges):
        # Pad cube to max spatial dimensions
        padded_cube = np.pad(cube, 
                            ((0, 0), 
                             (0, max_y - cube.shape[1]), 
                             (0, max_x - cube.shape[2])),
                            mode='constant')
        
        # Place in supercube
        supercube[start_idx:end_idx+1] += padded_cube
    
    return supercube

def plot_results(cubes, aligned_images, resampled_cubes, supercube, names):
    """Create diagnostic plots"""
    n_cubes = len(cubes)
    if n_cubes == 0:
        return
    
    # Plot original and aligned white images
    plt.figure(figsize=(15, 10))
    
    # Original white images
    for i in range(min(n_cubes, 3)):
        plt.subplot(2, 3, i+1)
        plt.imshow(np.nansum(cubes[i], axis=0), origin='lower', cmap='viridis')
        plt.title(f'Original {names[i]}')
    
    # Aligned white images
    for i in range(min(n_cubes, 3)):
        plt.subplot(2, 3, i+4)
        plt.imshow(aligned_images[i], origin='lower', cmap='viridis')
        plt.title(f'Aligned {names[i]}')
    
    # Supercube white image
    plt.subplot(2, 3, 3)
    plt.imshow(np.nansum(supercube, axis=0), origin='lower', cmap='viridis')
    plt.title('Supercube White Image')
    
    # Example spectrum
    plt.subplot(2, 3, 6)
    y, x = supercube.shape[1]//2, supercube.shape[2]//2
    plt.plot(supercube[:, y, x], label='Supercube Spectrum')
    plt.xlabel('Wavelength index')
    plt.ylabel('Flux')
    plt.title(f'Spectrum at ({x}, {y})')
    
    plt.tight_layout()
    plt.show()
    

def save_supercube(supercube, new_wave, headers, path, original_names, galname):
    """
    Save the supercube with all wavelength info in primary header
    """
    # Create new primary HDU
    hdu = fits.PrimaryHDU(supercube)
    new_header = hdu.header
    
    if headers:  # Only copy WCS if we have headers
        # Copy essential WCS from first header
        for key in ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2']:
            if key in headers[0]:
                new_header[key] = headers[0][key]
        
        # Add observation info
        for key in ['TELESCOP', 'INSTRUME', 'OBJECT', 'DATE-OBS', 'EXPTIME']:
            if key in headers[0]:
                new_header[key] = headers[0][key]
    
    # Add wavelength calibration
    new_header['NAXIS'] = 3
    new_header['NAXIS1'] = supercube.shape[2]
    new_header['NAXIS2'] = supercube.shape[1]
    new_header['NAXIS3'] = supercube.shape[0]
    
    if headers:
        new_header['CD1_1'] = headers[0].get('CD1_1', 1.0)
        new_header['CD2_1'] = headers[0].get('CD2_1', 0.0)
        new_header['CD1_2'] = headers[0].get('CD1_2', 0.0)
        new_header['CD2_2'] = headers[0].get('CD2_2', 1.0)
    
    new_header['CTYPE3'] = ('WAVE', 'Wavelength in Angstroms')
    new_header['CUNIT3'] = ('Angstrom', 'Wavelength units')
    new_header['CRPIX3'] = (1, 'Reference pixel')
    new_header['CRVAL3'] = (new_wave[0], 'Starting wavelength [Angstrom]')
    new_header['CDELT3'] = (np.mean(np.diff(new_wave)), 'Wavelength step [Angstrom]')
    
    # Store wavelength array as header cards (chunked if necessary)
    max_comment_len = 68  # FITS header comment max length
    chunk_size = 10  # Number of values per card
    for i in range(0, len(new_wave), chunk_size):
        card_name = f'WAVE{i//chunk_size:04d}'
        values = new_wave[i:i+chunk_size]
        comment = f'Wavelengths {i} to {min(i+chunk_size, len(new_wave))-1}'
        new_header[card_name] = (','.join(f'{v:.3f}' for v in values), comment)
    
    # Add processing history
    history = [
        'Supercube created by combining: ' + ', '.join(original_names),
        f'Wavelength range: {new_wave[0]:.1f}-{new_wave[-1]:.1f} Angstroms',
        f'Spectral sampling: {np.mean(np.diff(new_wave)):.3f} Angstroms/pixel',
        'Processing steps:',
        '1. Spatial alignment via cross-correlation',
        '2. Spectral resampling to common grid',
        '3. Data combination'
    ]
    for line in history:
        new_header.add_history(line)
    
    # Save to file
    output_filename = os.path.join(path, galname+'_supercube.fits')
    hdu.writeto(output_filename, overwrite=True)
    
    # Verification
    print(f"\nSaved supercube to {output_filename}")
    print(f"Wavelength range: {new_wave[0]:.1f} to {new_wave[-1]:.1f} Angstroms")
    print(f"Header contains {len(new_wave)} wavelength points in WAVE* keywords")


def get_center_from_header(header, original_shape, max_shape, shift):
    """
    Extract center pixel coordinates from FITS header and adjust for processing
    
    Parameters:
    header: FITS header
    original_shape: (y, x) shape of the original image
    max_shape: (y, x) maximum shape after padding
    shift: (dy, dx) shift applied during alignment
    """
    try:
        # Get original center from header
        if 'CRPIX1' in header and 'CRPIX2' in header:
            center_x = header['CRPIX1'] - 1  # Convert to 0-indexed
            center_y = header['CRPIX2'] - 1  # Convert to 0-indexed
        else:
            # Use image center as fallback
            center_x = original_shape[1] / 2 - 0.5
            center_y = original_shape[0] / 2 - 0.5
        
        print(f"Original center: x={center_x:.1f}, y={center_y:.1f}")
        
        # Apply the shift that was applied during alignment
        center_x += shift[1]  # dx component
        center_y += shift[0]  # dy component
        
        print(f"After shift: x={center_x:.1f}, y={center_y:.1f}")
        
        return center_x, center_y
        
    except Exception as e:
        print(f"Error getting center from header: {e}")
        # Fallback to center of padded image
        return max_shape[1] / 2 - 0.5, max_shape[0] / 2 - 0.5


def main(galname,centerpix):
    global supercube
    try:
        # Load data
        path = '.'
        cubes, headers, names = load_datacubes()
        
        # Get user-specified transformations
        transformations = get_transformations(names)
        
        # Get wavelength information
        wave_info = get_wavelength_info(headers)
        for name, info in zip(names, wave_info):
            print(f"{name}: {info['min']:.1f}-{info['max']:.1f} angstrom, step={info['step']:.2f} angstrom")
        
        # Create white light images for alignment
        white_images = create_white_images(cubes)
        
        # Align images with transformations
        aligned_images, shifts, max_shape = align_images(white_images, names, transformations)
        
        # print(shifts)
        # shifts[1] = np.array([0,0])
        
        # Apply shifts to full datacubes with transformations
        shifted_cubes = apply_spatial_shifts(cubes, shifts, max_shape, names, transformations)
        
        # Get center coordinates from first header, adjusted for processing
        center_coords = get_center_from_header(
            headers[0], 
            original_shape=white_images[0].shape,  # Original image shape
            max_shape=max_shape,                   # Padded shape
            shift=shifts[0]                        # Shift applied
        )
        
        # Resample spectra to common wavelength grid
        resampled_cubes, new_wave, wave_ranges = resample_spectra(shifted_cubes, wave_info)
        
        # Create supercube using the wavelength ranges
        supercube = create_supercube(resampled_cubes, new_wave, wave_ranges)
        
        # Create white light image from supercube
        white_image = np.nansum(supercube, axis=0)
        
        # Analyze galaxy
        cx, cy, r_half = analyze_galaxy(supercube, new_wave, white_image, path, center=centerpix)
        
        print("\nVerifying wavelength solution:")
        for i, info in enumerate(wave_info):
            print(f"Original {names[i]} range: {info['min']:.1f}-{info['max']:.1f} Å")
        print(f"Supercube range: {new_wave[0]:.1f}-{new_wave[-1]:.1f} Å")
        
        # Plot results
        plot_results(cubes, aligned_images, resampled_cubes, supercube, names)
        
        # Save the supercube to FITS file
        save_supercube(supercube, new_wave, headers, path, names, galname)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main(galname,centerpix)