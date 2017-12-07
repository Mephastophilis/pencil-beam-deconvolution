# pencil-beam-deconvolution
Diffuse optical X-ray-induced luminescence deconvolution with a pencil beam geometry

XL_PBXIL_decon_parallel.py runs a deconvolution based on a pencil beam geometry on the X-ray induced luminescence images collected with an EMCCD. The code is parallelized so that it runs the deconvolution on multiple images corresponding to various pencil beam positions. The deconvolution is based on a Richardson Lucy algorithm contained in PBXIL_Richard_Lucy.py. The final output of XL_PBXIL_decon_parallel.py is the individual deconvolutions for each pencil beam position and a reconstruction of the entire slice in the object that the pencil beams occupy.
