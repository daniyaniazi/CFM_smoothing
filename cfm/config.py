autoencoder_input_dim_dict = {'clip_ViT-B16_out': 512,  
                              'dinoclip_ViT-B16' : 512,
                              'dinoclip_openai_ViT-B16': 512,
                              }

# paths for the MPI cluster
data_dir_root = '/scratch/inf0/user/kwittenm/data'  # Kai's pre-computed features
save_dir_root = '/scratch/inf0/user/kwittenm/SAE'  # Kai's SAE checkpoints here
probe_cs_save_dir_root = '/scratch/inf0/user/kwittenm/probe'  # Kai's linear probes here
vocab_dir = '/scratch/inf0/user/kwittenm/vocab'
analysis_dir = '/BS/dniazi_thesis/work/CFM_smoothing/analysis'

# Override path for concept_names.txt 
concept_names_override = '/BS/dniazi_thesis/work/cfm_data/concept_names.txt'

probe_dataset_root_dir_dict = {
    "places365": "/BS/CC3M/static00/Places365Standard", 
    "imagenet": "/scratch/inf0/user/mparcham/ILSVRC2012", 
    "coco_stuff": "/BS/databases15/coco_stuff164", 
    "coco": "/BS/databases15/coco_stuff164",  # COCO images
    "cc12m": "/BS/databases32/cc12m_wds", 
    "cityscapes": "/BS/databases15/cityscapes_release", 
    "cc3m": "/BS/CC3M/static00/CC3M_TAR", 
}

probe_dataset_nclasses_dict = {"places365": 365, 'imagenet': 1000, "coco_stuff": 171,
                                "coco": 80, "cityscapes": 19, "cc3m": 1, "cc12m": 1}

config_dir = "cfm/clip_dinoiser_backbone/configs"