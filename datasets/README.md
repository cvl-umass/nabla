# NABLA Evaluation Datasets

This directory contains the paired image datasets used for evaluating identity-preserving generation in the paper: **"Not All Birds Look The Same: Identity-Preserving Generation For Birds"**. 

Because standard datasets like NABirds can contain pairs with significantly different plumages or dimorphism under the same class label, we provide these curated datasets to ensure accurate evaluation of identity preservation.

## Datasets Overview

We provide three datasets of image pairs, representing both in-domain and out-of-domain species evaluations:

1. **NABLA (`nabla_pairs.csv`)**: A rigorously filtered subset of the NABirds dataset. This dataset ensures that the paired images share the same specific identity and plumage, mitigating the evaluation inaccuracies caused by dimorphism or age-related plumage differences in the base NABirds dataset.
2. **iNat-Seen (`inat_nabirds_pairs.csv`)**: A dataset sourced from iNaturalist observations, containing image pairs for bird species that overlap with the training distribution (species seen during training).
3. **iNat-Unseen (`inat_unseen_pairs.csv`)**: A dataset sourced from iNaturalist, containing image pairs for bird species that were strictly *unseen* during training, used to evaluate the zero-shot identity preservation capabilities of the model.

## Data Preparation
1. Download the NABirds dataset from [here](https://dl.allaboutbirds.org/nabirds).
2. Extract the images into `data/nabirds/images/`.
3. Ensure the structure matches the paths provided in `datasets/nabla_pairs.csv`.
4. Download the iNaturalist observations using their [API](https://api.inaturalist.org/v2/docs/).

---

## File Structures

### 1. NABLA Pairs (`nabla_pairs.csv`)
Contains the curated pairs from the NABirds dataset.
* `class_id`: The NABirds class identifier (e.g., `0295`).
* `image_1`: Relative path to the first image in the pair (e.g., `./images/0295/...jpg`).
* `image_2`: Relative path to the look-alike image.

### 2. iNat-Seen Pairs (`inat_nabirds_pairs.csv`)
Contains pairs from iNaturalist for species seen during training.
* `observation_id`: Unique iNaturalist observation identifier.
* `common_name` / `scientific_name`: The common and scientific names of the bird.
* `relative_path`: The base directory path for the observation images.
* `observed_on`: Date and time of the observation.
* `photo_count`: Number of photos in the original observation.
* `latitude` / `longitude`: Geographic coordinates of the observation.
* `user_login`: iNaturalist username of the observer.
* `inat_url`: Direct link to the iNaturalist observation page.
* `query_species`: The species name used to query the data.
* `image_1`: Filename of the first image in the pair.
* `image_2`: Filename of the second image in the pair.

*(Note: To construct the full image path, combine `relative_path` + `/` + `image_1` or `image_2`).*

### 3. iNat-Unseen Pairs (`inat_unseen_pairs.csv`)
Contains pairs from iNaturalist for species unseen during training. 
* Contains the exact same iNaturalist metadata columns as the seen dataset (`observation_id`, `common_name`, `scientific_name`, `latitude`, `longitude`, etc.), with the exception of the `query_species` field. `image_1` and `image_2` denote the filenames of the image pair.