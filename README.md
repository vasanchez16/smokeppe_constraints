# smokeppe_constraints
Implementation of frequentist confidence sets for climate model parameter constraints

# Set-up

From the root directory, run the `run.py` file as a module as follows:

```
python -m run --input_file <path-to-input-json-file> --output_dir <path-to-output-directory> --savefigs
```

If desired, set up a default input and output with a `config.ini` file. For example, create the `.ini` file in the `script` subdirectory as follows:

```
[DEFAULT]
InputFile = /jet/home/jcarzon/smokeppe_constraints/examples/test_params.json
OutputDir = /ocean/projects/atm200005p/jcarzon/results/
```
