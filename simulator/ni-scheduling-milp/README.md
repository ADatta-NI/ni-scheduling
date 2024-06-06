## Project Setup

1. Install python version `3.10.13`
2. Clone the repository
   ```
   git clone https://github.com/bhanuvikas/ni-scheduling-milp.git
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Enter the repo
   ```
   cd ni-scheduling-milp
   ```
4. Execute the below command to understand the command-line options available for experimentation
   ```
   python scheduling_milp.py --help
   ```
5. Sample Run Command
    ```
    python scheduling_milp.py -s SCIP -th 1 -scf "data/small_sc.json" -rmg 10 -tl 300
    ```
   The above command runs the MILP scheduling formulation on the problem defined in "data/small_sc.json" file using the SCIP solver and 1 concurrent execution threads with a time limit of 300 seconds and relative mip gap configured as 10