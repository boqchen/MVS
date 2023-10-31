## General information
Almost all experiments consist of a main script and a network definition. The main script usually contains a dataset definition that suits the particular requirements of the respective experiment. Experiments contain an entry point (meaning they can be started as a Python script) and usually allow a range of caller arguments.

## Description of each experiment
- ae: Holds code for autoencoding experiments. Simple per-cell MLP as well as a convolutional AE that works with raw IMC images. Hasn't been run in quite some time.
- bulk: Holds code for bulk experiment. Really outdated and unnecessary. Can probably be deleted.
- celltype: Holds code for cell type classification. Main script holds code for simple MLP, random_forest.py contains a helper function for RF inference.
- cgan: Graveyard for old translator versions.
- cgan2: Misleading name, should be called translator. Holds the current version of the translator training. Recommend that you read it.
- he2imc: Holds all code for the HE2IMC UNet
- imc2he: Holds all code for the IMC2HE UNet
- imcpatch: Holds all code for the IMCPatch network
- turmorpatch: Holds all code for the TumorPatch network
- wganimc: Holds code for a IMC-only WGAN. Didn't work at all. Really outdated. Can probably be deleted.
- inference.py: script to run predictions on the test set (common to all models)
