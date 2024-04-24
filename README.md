# smokeppe_constraints

Implementation of frequentist confidence sets for climate model parameter constraints

# Contents

- `examples`: Files in JSON format instantiating parameters and data domains for constraint examples.
- `script`: The subdirectory from which methods should be interface by the command line.
- `src`: Methods to be called indirectly through `script`.

## Use

From the  `script` directory, run the `run.py` file as a module as follows:

```
python -m run --input_file <path-to-input-json-file> --output_dir <path-to-output-directory> --savefigs
```

If desired, set up a default input and output with a `config.ini` file. For example, create an `.ini` file in the `script` subdirectory as follows:

```
[DEFAULT]
InputFile = /input/directory/evalParameters.json
OutputDir = /output/directory/
```

See `examples/evalParametersTemplate.json` for example of the contents of this json file.

## Options

Our method calls for several options for the noise model, each requiring bespoke estimation methods.

### Gaussian noise model

The Gaussian noise model simply optimizes the closed form likelihood with the `scipy` implementation of `LBFG-S`.

### Student-t noise model

This model optimizes two values for the student-t approximation, shape $\nu$ and scale $\delta$. For numerical optimization, bounds on the search range will be included in the JSON example configuration file where the first set of bounds will correspond to $\delta_{MLE}$ and the second set of bounds will correspond to $\nu$. It is required mathematically that $\nu>2$ and that $\delta>0$.
