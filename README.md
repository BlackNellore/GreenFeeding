# GreenFeeding
Multiobjective diet optimization model for beef cattle based on "Nutritional Requirements for Beef Cattle" 8th Ed. (NASEM, 2016) and RNS 1.0 (Tedeschi and Fox, 2020).

This model is build on top of a profit-maximing optimization model (Marques et al, 2020). It takes available feedstuff (price, min, and max inclusion) and animal's characteristics, and ingredients life cycle assessment (LCA) of different environmental impacts to compute a multiobjective efficiency frontier.
The model can use either feedstuff properties inputed directly or use the RData file generated by the [RNS software](https://www.nutritionmodels.com/rns.html). Moreover, methane (CH4) and nitrous oxide (N2O) emissions are computed using Escobar-Bahamondes et al. (2017) and the IPCC (2006), respectively.

Always consult a veterinary before changing the feed composition for your herd.

This model is part of a PhD project at the University of Edinburgh and a thematic project funded by CAPES.

### Publication:
Marques, J.G.O., de O. Silva, R., Barioni, L.G., Hall, J.A.J., Fossart, C., Tedeschi, L.O., Garcia-Launay, F., Moran,
D., 2021.
[Evaluating environmental and economic trade-offs in cattle feed strategies using multiobjective optimization](https://doi.org/10.1016/j.agsy.2021.103308).
Agricultural Systems.

### Citation:
Marques, J.G.O., 2021. GreenFeeding [WWW Document]. URL https://github.com/BlackNellore/GreenFeeding.


#### References

* Escobar-Bahamondes, P., Oba, M., Beauchemin, K.A., 2017. Universally applicable methane prediction equations for beef cattle fed high- or low-forage diets. Can. J. Anim. Sci. 97, 83–94. https://doi.org/10.1139/cjas-2016-0042
* IPCC, 2006. 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Directrices para los inventarios nacionales GEI. Institute for Global Environmental Strategies, Hayama, Japan.
* Marques, J.G.O., de O. Silva, R., Barioni, L.G., Hall, J.A.J., Tedeschi, L.O., Moran, D., 2020. An improved algorithm for solving profit-maximizing cattle diet problems. Animal 14, s257–s266. https://doi.org/10.1017/S1751731120001433
* NASEM - National Academies of Sciences, Engineering, and Medicine 2016. Nutrient Requirements of Beef Cattle, 8th Revised Edition. National Academies Press, Washington, D.C.
* Tedeschi, L.O., Fox, D.G., 2020a. The Ruminant Nutrition System, Volume I – An Applied Model for Predicting Nutrient Requirement and Feed Utilization in Ruminants, 3rd ed. XanDu Publishing, Inc., Ann Harbor, MI, USA.
* Tedeschi, L.O., Fox, D.G., 2020b. The Ruminant Nutrition System: Volume II - Tables of Equations and Coding. XanEdu, Ann Harbor, MI, USA.


## Getting Started
### Running on Docker:
####Requisites:
* [Docker](https://www.docker.com/products/docker-desktop)
* Approximately 1.5 Gb free space to build the image

#### Configuration
1. Download/Clone the project in a local folder.
2. Open a terminal in the folder containing the project.
   ```bash
   >cd [project_path_in_double_quotes]
   ```
3. Build the docker image. Note the full stop "." at the end of the command, this will build the file "Dockerfile"
   inside the project folder.
    ```bash
   >docker build -t greenfeeding .
   ```
4. Run the docker image.
    ```bash
   >docker run --rm --name gf_run -it greenfeeding /bin/bash
   ```
5. Run the optimization model.
    ```bash
   >python3 run.py
   ```
6. Get results out of Docker. Without closing (or exiting) the first terminal, open a **second terminal** on the same
   folder and run:
    ```bash
    >docker cp gf_run:/Output .
   ```

7. To run different scenarios, you can change the file "input.xlsx" and copy it inside the container on the
   **second terminal**:

   Terminal 2: 
   ```bash
    >docker cp ./Input.xlsx gf_run:/.
   ```
   Terminal 1:
   ```bash
   >python3 run.py 
   ```

Note: the option ```--rm``` in step 4 will delete the container after you exit it. Setup your docker container your 
preferred way by adjusting these options.

### Running Locally
#### Requisites:
* [Python](https://www.python.org/downloads/) 3.7 or later. Note: CPLEX compatible with 3.8 and below.
* Linear programming solver supported by [Pyomo](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers)
* [R](https://www.r-project.org/)
* Python packages:
    * numpy
    * xldr
    * openpyxl
    * aenum
    * pandas
    * scipy
    * rpy2 (requries R)
    * pyomo
    * tqdm

These can be installed by executing on the local folder terminal:
```pip3 install -r requirements``` or ```pip install -r requirements```.

### Run and Setup
Executing the optimization on the local folder terminal (or IDE of your choice):
```
>python run.py
```
Important files:
* Input in **"./input.xlsx"**
* Ouput in **"./Output/[timestamp]"**
* Log in **"./activity.log"**
* Settings in **"./config.py"**

After each run, the input used and the csv containing the output are saved in the folder "/Output".

### Basic Settings
1. Adjust your input in the file **"./input.xlsx"**:
    1. Sheet "Feeds": Choose the available feeds setting the ID, it will automatically retrieve the name from sheet
       "FeedLibrary" (NASEM, 2016).
       Set minimum and maximum concentration allowed (between 0 and 1), and feed cost \[US$/kg\].
       The column "Feed Scenario" aggregates all feedstuff that belong to a particular scenario.
       This index will be matched with the one in the sheet "Scenario"
    2. Sheet "Scenario":
        * ID: Scenario ID \[int\]
        * Feed Scenario: Define which feed scenario should be match in the sheet "Feeds" \[int\]
        * Batch: ID crossed with table Batch
        * Breed: Breed Name (does not affect result)
        * SBW: Shrunk Bodyweight \[100; 800\]
        * BCS: Body Condition Score \[0; 9\]
        * BE: Breed Factor (check NASEM 2016, pg355 Table 19-1)
        * L: Lactation Factor {1, 1.2}
        * SEX: {1, 1.15}
        * a2: 0 if not considering acclimatization factor, check NASEM (2016) otherwise
        * PH: Rumen desired pH
        * Selling Price: Cattle Selling Price per \[U$/kg\]
        * Linearization factor: an coefficient to adjust nonlinear SWG by a line.
        * Algorithm: BF - Brute Force; GSS - Golden Section Search
        * Identifier: String to name sheets when writing results
        * LB: Concentration of Net Energy for Maintenance (CNEm) \[Mcal/kg\] lower bound (suggestion: 0.8)
        * UB: Concentration of Net Energy for Maintenance (CNEm) \[Mcal/kg\] upper bound (suggestion: 3.0)
        * Tol: Result tolerance (suggested: 0.01)
        * Obj: MaxProfit (maximizes profit), MinCost (minimizes cost), or MaxProfitSWG (maximize profit/shrunk weight
          gain)
2. Run:
    ```
    >python run.py
    ```
3. Results: if everything is alright, you can check your solution on **"./output.xlsx"**. Otherwise, you can check the
   **"./activity.log"** to see if any errors happened.

## Advanced settings
### Options:
<img src="/Doc/Advanced.drawio.png">

### Batch Run
1. The table 'Batch' takes 5 inputs:
    * Batch ID: ID to be crossed with table 'Scenario'
    * Filename: path + filename of CSV file with batch info
    * Period col: name of the col that contains the IDs of each running
    * Initial period: p<sub>i</sub>  such as 'Period col' &ge; p<sub>i</sub>
    * Final period: p<sub>f</sub>  such as 'Period col' &le; p<sub>f</sub>
2. Example CSV file 'test.csv':

   | row_ids   | DDGS_01      | Animal_price   | ...   | variable_m    |
       | ----------|:-------------:|:-------------:| :----:|:-------------:|
   | 1         | 0.86          | 5.43          | ...   | 17.4          |
   | 2         | 1.23          | 3.45          | ...   | 13.2          |
   | ...       |...            | ...           | ...   | ...           |
   | n         | 2.26          | 4.25          | ...   | 11.9          |
3. For the table above 'Filename' = 'test.csv' and 'Period col' = 'row_ids'. We could also set 'Initial period' = 2 and 'Final period' = 7 or leave
   them blank to use the whole file.
4. The following tables and columns can be have the values replaced for a batch column:
    1. Feeds:
        * Min %DM
        * Max %DM
        * Cost [US$/kg AF]
    2. Scenario
        * SBW
        * BCS
        * BE
        * L
        * SEX
        * a2
        * PH
        * Selling Price [US$]
        * Linearization factor
5. To run the batch simply make sure that you place the name of the batch column in the cell that you want to run as a
   batch. For example, instead of putting a value in the column 'Selling Price [US$]' on table 'Scenario', one could write
   'Animal_price', assuming the file showed on item 2.

NOTE: If a scenario has batch ID = -1 or blank, i.e., it is not a batch scenario, having strings in place of values
will raise error. So pay attention if you have multiple scenarios and not all are batch.

### Reduced Cost Search:

In the sheet "Scenario", set "Find Reduced Cost" to 0 or -1 to disable the function. Otherwise, set it to the
ingredient ID accordingly to sheet "Feeds". In "Ingredient Level", define the threshold inclusion used to find the
reduced cost, i.e., we changed the traditional reduced cost search in linear programming, which finds the basic
solution where x >= 0, replacing 0 with the desired inclusion level defined. Since the model is nonlinear, finding
this value is an iterative process that uses the bissection search method, hence using zero or any other value does
not affect performance.

NOTE: if the option in "config.py" is defined as```RNS_FEED_PARAMETERS = {'source': 'RNS/image.RData'```, the
ingredient must be created in the [RNS](https://www.nutritionmodels.com/rns.html) and the ID must match that of the
exported ingredient in the image.RData file. if the 'source' is defined as ```RNS_FEED_PARAMETERS = {'source':
None```, the ID and ingredient's properties used are those from the Excel sheet "Feed Library".


## Settings
You can change the file names and other settings in ```config.py```.
Be sure to have headers and sheet names matching in the ```config.py``` and ```input.xlsx```.
```python
SOLVER = 'glpk' ## Use keyword defined by the Pyomo library
RNS_FEED_PARAMETERS = {
    'source': 'RNS/image.RData', ## Uses RNS generated file
    # 'source': None, ## Uses properties on Exel spreadsheet
    'report_diff': False,
    'on_error': 0}  # 0: quit; 1: report and continue with NRC; -1: silent continue;
INPUT_FILE = {'filename': {'name': 'iInput.xlsx'},
              'sheet_feed_lib': {'name': 'Feed Library',
                                 'headers': [...]},
              'sheet_feeds': {'name': 'Feeds',
                              'headers': [...]},
              'sheet_scenario': {'name': 'Scenario',
                                 'headers': [...]}}
```
