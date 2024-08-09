This project contains multiple versions of data generators, utils and training and evaluation files. This version is tailored to fit manufacturing data to our inference logic. Please try inference with only manufacturing data named "static_configuration_plan" 
The trained agents are trained without taking priority into consideration. Since Manufacturing data had a dynamic priority assignment, i have avoided priority for the final manufacturing data results. 
This master version is highly modified to accomodate manufacturing data. 
Another branch with no modifications for manufacturing data will also be released shortly. 


## Project Setup

1. Install python version `3.11`
2. Clone the repository
   ```
   
   ```
3. Enter the repo
   ```
   cd ni-scheduling-project
   ```
4. Create and activate a python virtual environmebnt
   ```
   python -m venv .venv

5. Use any of the trained agents to infer using the static configuration files in intermediate_files folder in the data section
   using evaluator.py. Uncomment functions like compute tardiness per product only if there are non zero planned quantitites.

6. Change 'planned_quantity' to 'job_quantity'  in data_generator_real.py  file if you want to schedule all items in a given order.

7. Change the simplex.py file in RLLib utils -> spaces -> simplex to make rllib compatible with gymnasium in the following manner
   ```
   def __init__(self, shape, concentration=None, dtype=np.float32):
        assert type(shape) in [tuple, list]

        super().__init__(shape, dtype)
        self.dim = self.shape[0]

        if concentration is not None:
            assert (
                concentration.shape == (shape[0],)
            ), f"{concentration.shape} vs {shape[0]}"
            self.concentration = concentration
        else:
            self.concentration = np.array([1] * self.dim)

    def sample(self):
        return np.random.dirichlet(self.concentration, size=self.shape[0]).astype(
            self.dtype
        )

