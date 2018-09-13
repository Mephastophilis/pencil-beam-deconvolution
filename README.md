# pencil-beam-deconvolution
Diffuse optical X-ray-induced luminescence deconvolution with a pencil beam geometry

Simulated_PB_decon_parallel.py runs a deconvolution based on a pencil beam geometry on the X-ray induced luminescence images collected with an EMCCD. The code is parallelized so that it runs the deconvolution on multiple images corresponding to various pencil beam positions. The deconvolution is based on a Richardson Lucy algorithm contained in PB_XLXF_Richard_Lucy.py. Requires a x-ray fluorescence image that serves as a binary mask for the x-ray luminsecence deconvolution. Requires the x-ray luminescence data as well.
