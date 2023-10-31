import os
import pandas as pd
import numpy as np
import json
import random
import cv2
import math
import torch
from pathlib import Path

if not __package__:  # looks like we are running this as a script, try simple import
    from raw_utils import get_manually_aligned_he_region, read_raw_protein_txt, get_rois_names_and_coords
    from constants import *
else:  # looks like we are running this as a module, try relative import
    from codebase.utils.raw_utils import get_manually_aligned_he_region, read_raw_protein_txt, get_rois_names_and_coords
    from codebase.utils.constants import *


def scan_patients_and_samples(use_cached=False, store_cached=False, verbose=0):
    ''' This function scans the dataset directory structure to build a map between patients and lists of samples. It utilizes caching to decrease runtime.
    use_cached: Whether or not to use any existing cached maps.
    store_cached: Whether of not to store the results of the respective call as a cached file. This will overwrite any pre-existing cached files.
    verbose: Level of verbosity
    '''
    if use_cached:
        # check if there are cached versions of the maps, saves time
        surrounding_dir = os.path.dirname(os.path.realpath(__file__))  # this is the directory that this file is stored in
        cached_files = [f.name for f in os.scandir(surrounding_dir) if f.is_file() and f.name.startswith("cached_maps_") and f.name.endswith(".json")]
        if "cached_maps_p2s.json" in cached_files and "cached_maps_s2p.json" in cached_files:
            if verbose >= 1:
                print("Using cached json files.")

            with open(os.path.join(surrounding_dir, "cached_maps_p2s.json"), 'r') as fh1, \
             open(os.path.join(surrounding_dir, "cached_maps_s2p.json"), 'r') as fh2:
                patient_to_samples = json.load(fh1)
                sample_to_patient = json.load(fh2)
                return patient_to_samples, sample_to_patient


    excl_patients = ["TP-A1-RT-001"]  # this folder looked weird, sample didn't have regular naming scheme
    excl_patients_path = [os.path.join(ROOT_DIR_STUDY, ep) for ep in excl_patients]

    patients = [(f.path, f.name) for f in os.scandir(ROOT_DIR_STUDY) if f.is_dir() and f.path not in excl_patients_path]
    
    # map patient->[samples] and sample->patient
    patient_to_samples, sample_to_patient = dict(), dict()

    for p in patients:
        # assuming that every subfolder inside a patient folder is a sample
        samples = [(f.name) for f in os.scandir(p[0]) if f.is_dir()]  # p[0] is path
        patient_to_samples[p[1]] = samples  # p[1] is patient id
        for s in samples:
            sample_to_patient[s] = p[1]

    if store_cached:
        if verbose >= 1:
            print("Writing new cached json files.")

        p2sjson = json.dumps(patient_to_samples)
        s2pjson = json.dumps(sample_to_patient)
        surrounding_dir = os.path.dirname(os.path.realpath(__file__))  # this is the directory that this file is stored in
        with open(os.path.join(surrounding_dir, "cached_maps_p2s.json"), 'w') as fh1, \
         open(os.path.join(surrounding_dir, "cached_maps_s2p.json"), 'w') as fh2:
            fh1.write(p2sjson)
            fh1.close()
            fh2.write(s2pjson)
            fh2.close()

    return patient_to_samples, sample_to_patient


def filter_and_transform(samples: list, ignore_tif=False):
    ''' This function expects a list of sample names and transforms it into a list of dictionaries.
    Each dictionary stores absolute paths for a WSI file, a JSON file with ROI coordinates and a HDF file with derived cell info.
    If any problems occur during the transform process, the respective sample is marked as problematic. The final returns of this function
    are two lists of usable samples and problematic_samples. The latter can be (or should be) safely ignored.
    samples: A list of sample names, e.g. ['MACEGEJ', 'MACOLUD'] or M41_SAMPLES
    ignore_tif: Whether or not to ignore .tif WSI files, as they seem to be stored with a different rotation (as opposed to .ndpi files)
    '''
    # obtain sample-to-patient maps:
    patient_to_samples, sample_to_patient = scan_patients_and_samples(use_cached=True)
    # prepare result lists:
    usable_samples = []
    problematic_samples = []

    for sample in samples:
        # [1] Get ROI coordinate data from json file ----------------------------------------------------------------
        try:
            # every sample folder in imc_he_rois/study/melanoma should have one .json file with ROI coordinates
            json_files = [f.name for f in os.scandir(os.path.join(ROOT_DIR_ROI_COORDS, sample)) if f.is_file() and f.name.endswith(".json")]
        except FileNotFoundError:
            problematic_samples.append((sample, "sample folder not found in datadrop ROI coordinate dir"))
            continue

        if len(json_files) != 1:
            problematic_samples.append((sample, "more than one or no json file"))
            continue

        json_file = json_files[0]  # this is save due to the check above
        roi_coords_path = os.path.join(ROOT_DIR_ROI_COORDS, sample, json_file)


        # [2] Based on json file, try to find corresponding HE slide file -------------------------------------------
        # determine HE slide file name
        he_slide_name = json_file[:json_file.rfind(".json")]  # cut away the ".json"
        # currently the code only handles tif and ndpi
        if not (he_slide_name.endswith("tif") or he_slide_name.endswith("ndpi")):
            problematic_samples.append((sample, "weird WSI file format detected"))
            continue  # continue on outer for-loop

        if he_slide_name.endswith("tif") and ignore_tif:
            problematic_samples.append((sample, "tif file filtered (ignore_tif=True)"))
            continue

        # determine highest pass:
        try:
            patientID = sample_to_patient[sample]
        except KeyError:
            problematic_samples.append((sample, "Sample not found in patient-sample-map"))
            continue

        patient_digpath_raw = os.path.join(ROOT_DIR_STUDY, patientID, sample, "digpath_zurich/raw/")
        try:
            passes = [f.name for f in os.scandir(patient_digpath_raw) if f.is_dir() and f.name.startswith("pass")]
        except FileNotFoundError:
            problematic_samples.append((patientID + "/" + sample, "patient/sample/digpath_zurich/raw/ not found"))
            continue

        # start from highest pass and see if we can find a matching HE slide file
        passes.sort(reverse=True)
        used_pass = ""
        he_slide_path = ""
        found_match = False
        for passn in passes:
            # check if corresponding HE slide file can be found
            he_slide_path = os.path.join(patient_digpath_raw, passn, he_slide_name)
            if os.path.exists(he_slide_path):
                used_pass = passn
                found_match = True
                break  # break out of inner for-loop

        if not found_match:
            problematic_samples.append((sample, "matching HE slide could not be found in any pass"))
            continue  # continue on outer for-loop


        # [3] Look for corresponding HDF file --------------------------------------------------------------------------------------
        available_hdfs = [os.path.splitext(f.name)[0] for f in os.scandir(FILTERED_SCE_STORAGE) if f.is_file() and f.name.endswith(".hdf5")]

        if sample not in available_hdfs:
            problematic_samples.append((sample, "No HDF file found in filtered SCE storage !"))
            continue

        sce_filepath = os.path.join(FILTERED_SCE_STORAGE, sample + ".hdf5")

        # finally, we can be relatively certain that this sample is usable
        usable_samples.append({"json_file": roi_coords_path,
                               "he_slide": he_slide_path,
                               "sce_hdf": sce_filepath,
                               "sample": sample,
                               "patient": patientID})

    return usable_samples, problematic_samples


def clean_train_val_test_sep_for_manual_aligns_ordered_by_quality(report_set=True, val_part=0.05, test_part=0.05):
    ''' This function creates a clean train-val-test separation for manual alignment results.
    report_set: If True, the data split used in the written report is returned. If false, it computes a new data split based on the list of manual alignment result files in the MANUAL_ALIGNS_STORAGE
    val_part: Validation part of the split.
    test_part: Test part of the split.
    '''
    manual_aligns_path = MANUAL_ALIGNS_STORAGE
    align_results_paths = [os.path.join(manual_aligns_path, f.name) for f in os.scandir(manual_aligns_path) if f.is_file() and f.name.endswith(".json")]

    align_results = []
    for arp in align_results_paths:
        with open(arp) as json_file:
            align_results.append(json.load(json_file))
            json_file.close()

    assert len([ar for ar in align_results if ar["precise_enough_for_l1"] and len(ar["good_areas"]) > 0]) == 0, "Inconsistency in align results !"

    if report_set:
        train_aligns = [ar for ar in align_results if ar["sample"] in TRAIN_SAMPLES]
        val_aligns = [ar for ar in align_results if ar["sample"] in VALID_SAMPLES]
        test_aligns = [ar for ar in align_results if ar["sample"] in TEST_SAMPLES]

        assert len([ar for ar in train_aligns if ar in val_aligns or ar in test_aligns]) == 0, "Data split not clean, train set"
        assert len([ar for ar in test_aligns if ar in train_aligns or ar in val_aligns]) == 0, "Data split not clean, test set"
    else:
        # compute data split with intended ordering
        # get list of all patients:
        patients = np.unique([ar["patient"] for ar in align_results])

        # the strategy now is to order the patients by decreasing quality: first samples with precise ARs, then those that at least have some good areas, then the rest.
        # the train set is then chosen from the start of the list, and the val/test set from the back, ensuring that we get the maximum of good data for train set.
        # use patients IDs, so we ensure that no leaking is happening
        patients_with_precise_rois = []
        patients_no_precise_rois_but_good_areas = []
        patients_imprecise = []  # the *alignments* are still good, the morphological differences are simply very large for this set.

        for p in patients:
            ars_for_patient = [ar for ar in align_results if ar["patient"] == p]

            if any([ar["precise_enough_for_l1"] for ar in ars_for_patient]):
                patients_with_precise_rois.append(p)
            elif any([len(ar["good_areas"]) > 0 for ar in ars_for_patient]):
                patients_no_precise_rois_but_good_areas.append(p)
            else:
                patients_imprecise.append(p)

        patients_ordered_by_quality = patients_with_precise_rois + patients_no_precise_rois_but_good_areas + patients_imprecise
        assert len(patients) == len(patients_ordered_by_quality), "Fatal logical inconsistency !"

        # round up number of val/test patients
        num_val_patients, num_test_patients = int(val_part * len(patients) + 0.5), int(test_part * len(patients) + 0.5)

        train_end = len(patients) - num_val_patients - num_test_patients
        val_end = train_end + num_val_patients
        test_end = len(patients)

        train_patients = patients_ordered_by_quality[0: train_end]
        val_patients = patients_ordered_by_quality[train_end: val_end]
        test_patients = patients_ordered_by_quality[val_end: test_end]

        # finally, convert back to align results
        train_aligns = [ar for ar in align_results if ar["patient"] in train_patients]
        val_aligns = [ar for ar in align_results if ar["patient"] in val_patients]
        test_aligns = [ar for ar in align_results if ar["patient"] in test_patients]

    return train_aligns, val_aligns, test_aligns


def write_manually_rotated_rois_to_disk(manual_aligns_path=MANUAL_ALIGNS_STORAGE, save_path=BINARY_HE_ROI_STORAGE, overwrite=False):
    # name is a bit misleading, was focussing on the rotation at the time.
    # name should be write_manually_aligned_he_rois_to_disk
    align_results_paths = [os.path.join(manual_aligns_path, f.name) for f in os.scandir(manual_aligns_path) if f.is_file() and f.name.endswith("json")]
    align_results = []
    for arp in align_results_paths:
        with open(arp) as json_file:
            align_results.append(json.load(json_file))
            json_file.close()

    for ar in align_results:
        save_file = os.path.join(save_path, ar["sample"] + "_" + ar["ROI"] + ".npy")
        if not overwrite and os.path.exists(save_file):
            continue
        he_roi = get_manually_aligned_he_region(align_result=ar, level=0)
        he_roi = he_roi.astype(float) / 255.0
        he_roi = cv2.resize(he_roi, dsize=(4000, 4000))
        np.save(save_file, he_roi)


def write_raw_imc_txt_to_disk(samples: list, save_path=BINARY_IMC_ROI_STORAGE, overwrite=False):
    for s in samples:
        imc_raw_folder = os.path.join(ROOT_DIR_STUDY, s["patient"], s["sample"], "imc/raw/pass_1/")
        imc_raw_txt_candidates = [f.name for f in os.scandir(imc_raw_folder) if f.is_file() and f.name.endswith(".txt")]

        roi_name_to_coords = get_rois_names_and_coords(s["json_file"])
        roi_names_from_json = list(roi_name_to_coords.keys())
        roi_names_from_json = [rnfj for rnfj in roi_names_from_json if rnfj not in REF_ROIS]

        for roi_id in roi_names_from_json:
            roi_candidates = [os.path.join(imc_raw_folder, irtc) for irtc in imc_raw_txt_candidates if roi_id in irtc[irtc.rfind("__"):]]

            if len(roi_candidates) != 1:
                print("Problematic to find corresponding raw IMC file for sample ", s["sample"], " and ROI ", roi_id)
                continue  # continue on ROI ID loop

            save_file = os.path.join(save_path, s["sample"] + "_" + roi_id + ".npy")
            if not overwrite and os.path.exists(save_file):
                continue

            imc_raw_txt_file = roi_candidates[0]
            imc_roi_np = read_raw_protein_txt(imc_raw_txt_file, transformation=(lambda x: math.log(x + math.sqrt(x**2 + 1))))
            np.save(save_file, imc_roi_np)

            
# get patient-sample mapping:
def get_patient_samples(data_dir = '/cluster/work/tumorp/data_repository/', phase='study', indication=None):
    ''' Get dataframe with all patient/sample pairs for a given indication and project phase
    data_dir: root directory containing data
    phase: project phase {study, poc, post-poc}
    indication: cancer indication {None, melanoma, ovca, aml}
    '''
    
    patients = os.listdir(Path(data_dir).joinpath(phase))
    if indication is not None:
        if indication=='melanoma':
            patients = [x for x in patients if x[3]=='M']
        elif idication=='ovca':
            patients = [x for x in patients if x[3]=='G']
        elif idication=='aml':
            patients = [x for x in patients if x[3]=='A']
            
    patient_sample_dict = dict()
    for patient in patients:
        patient_sample_dict[patient] = pd.Series(os.listdir(Path(data_dir).joinpath(phase, patient)))
        
    patient_sample_df = pd.concat(patient_sample_dict).reset_index()
    patient_sample_df.columns = ['patient', 'idx', 'sample']
    patient_sample_df = patient_sample_df.loc[:,['patient', 'sample']]
    
    return patient_sample_df

