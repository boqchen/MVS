## Utility Code Collection
- automatic_alignment: Holds the code for the automatic alignment procedure. Make sure to read it before you run it, as you might overwrite results otherwise !
- clear_pycache: Simple script to delete all pycaches below the target directory.
- constants: Important collection of constants, including sample lists, path constants, value constants, name maps etc. Make sure to read this !
- dataset_utils: Holds code for dataset creation / filtering.
- derived_utils: Holds code for handling the derived SCE data.
- insert_cd31: Some raw IMC files are missing the CD31 channel. This script allows you to hackily insert the CD31 channel by computing the Voronoi regions based on the derived SCE data. This is not a very smart thing to do, as there is no accounting for cell distances, but better than missing CD31 entirely.
- mark_good_areas: Application that let's you manually mark good areas in ROIs. Becaus proper GUI programming is time-consuming it uses the simple matplotlib GUI API.
- marker_aligns: Application that let's you manually align ROIs by clicking on prominent markers. This program is a hot mess. It contains more bugs than the movie 'Antz'. If you ever need to use it, drop me a message.
- metrics: Small and simple collection of error metrics, such as MSE or MI
- raw_utils: Holds code for handling raw data, i.e. raw IMC images and HE images.
- spmap: Sample-Patient map. Let's you quickly retrieve a list of samples for a given patient, or a sample's corresponding patient.
- stainNorm_Macenko: External dependency. Holds code for stain normalization.
- stain_utils: External dependency. Holds code for stain normalization.
- zipper: Holds code that zips the entire codebase and timestamps it.
