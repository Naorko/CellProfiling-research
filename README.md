# CellProfiling-research

## Instructions
### Preperations
```bash
conda create --name tf-env python=3.8.6 tensorflow-gpu=2.2.0 scikit-learn pandas tqdm progressbar2 matplotlib openpyxl
```
> To create an enviroment named tf-env, need to be executed once.

If we want to be able to run a jupyter notebook server we will need to run `conda install --name tf-env jupyterlab`

-------------
### Execute a batch job
There are some batch files currently in our work folder
1. `sbatch-downloader` for downloading plates
2. `sbatch-run-diagnose-all` for running the main project for the big plates folder
3. `sbatch-run-diagnose-few`for running the main project for the smaller folder of plates named "few"
4. `sbatch-notebook` for running a jupyter notebook server

> In the sbatch files we can redefine various things like scripts' parameter, email notifications, etc.

In order to execute a job we need to do the following thing
`sbatch <sbatch-file-name>`

Extras:
- We can define a dependacy between excuting jobs like
`sbatch --dependency=afterok:<other_job_id> <sbatch-file-name>`
a possible usage for that is running with the big plate folder only after it was successfuly executed with the small folder

- At our lab we have a few "golden tickets" which will give us priority for nodes. We can use it wisely if we want to ran with less chance to be preempted.
`sbatch --qos=assafzar <sbatch-file-name>`

-------------

### Scripts
#### /code/learning_tabular_scaled/main.py 
Run models on the tabular data
##### Usages
- Usage: `main.py <parameters>` as configed in `/code/learning_tabular_scaled/configuration/config.py`

##### Examples
- Usage: `main.py` will run over with the default parameters

##### Output
Generates output in the given experiment directory

-------------
#### /code/learning_visual/main.py 
Run models on the visual data
##### Usages
- Usage: `main.py <parameters>` as configed in `/code/learning_visual/configuration/config.py`

##### Examples
- Usage: `main.py` will run over with the default parameters

##### Output
Generates output in the given experiment directory

-------------
#### /code/learning/main.py
##### Usages
- Usage 1: without parameters, will run over the default big directory
- Usage 2: `main.py <working_folder>` will run over the given directory

##### Examples
- Usage 1: `main.py` will run over with the default big plate folder - option will be removed in the future
- Usage 2: `main.py C:\plates` will run over C:\plates

Note: Currently the directory must contain the "csvs" folder from the downloader script

##### Output
Generates output in the given directory in a folder named "results"

-------------
#### /code/plateDownloader.py
##### Usages
- Usage 1: `plateDownloader.py <working_folder> -n <plate_amount>`
- Usage 2: `plateDownloader.py <working_folder> -l <plate_number1> <plate_number2> ...`
- Usage 3: `plateDownloader.py <working_folder> -n <plate_amount> -l <plate_number1> <plate_number2> ...`

##### Examples
- Usage 1: `plateDownloader.py C:\plates -n 5` will download 5 random plates from the available plates at the ftp server to C:\plates
- Usage 2: `plateDownloader.py C:\plates -l 26569 26572 25732` will download plates: 26569, 26572 and 25732 to C:\plates
- Usage 3: `plateDownloader.py C:\plates -n 2 -l 26569 26575 26574 26576` will download 2 random plates from the given plates' number to C:\plates

##### Notes:
- Will download files that not already exists in the given folders
- With the "n" parameter, selects n plates randomly from the given list or from the ftp server

##### Output
Generates three folders in the given directory
- tars: contain the raw tar.gz files of the plates
- extracted: the relevant files from the compressed files
- csvs: the relevant details for each plate in a csv file per plate, contain all the details for the cells from this plate.

-------------
#### /code/extractStatistics.py
##### Usages
- Usage: `extractStatistics.py <csv_folder> <output_folder>`

##### Examples
- Usage: `extractStatistics.py C:\plates\csvs C:\plates\stats` will run over the plates' csv files in C:\plates\csvs and create their statistics in C:\plates\stats

##### Notes:
- None

##### Output
Currently generates:
- Excel file with each plates' statistics.
- Plot for every feature in each plate - will be replaced in the future
