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
InputFile = /input/directory/evalParameters.json
OutputDir = /output/directory/
```

See evalParametersTemplate.json for example of the contents of this json file.

# Notes for MLE Optimization
Including proper bounds for the optimization is very important for the algorithm to converge accurately.

## Student-t approximation MLE
The algorithm optimizes two values for the student-t approximation. The two decision variables are the third standard deviation term, $\delta_{MLE}$, and the degrees of freedom for the student-t distribution, $\nu$. The bounds for the optimization algorithm should be included in the .json file where the first set of bounds will correspond to $\delta_{MLE}$ and the second set of bounds will correspond to $\nu$.

## Convolution
Add notes here