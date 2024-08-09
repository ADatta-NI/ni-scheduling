import json
from datetime import datetime, timedelta
import os

"""Below are the functions used sequentially to clean and convert NI hungary data to our data format
the output files are included in the folder intermediate files 
The main function is not given as these steps may not be required 
depending on data format used 
If required add the main function and execute the functions in the sequence
Change the path names accordingly to the intermediate files folder 


"""


def assign_values(config1_path, config2_path):
    with open(config1_path, 'r') as file1, open(config2_path, 'r') as file2:
        config1 = json.load(file1)
        config2 = json.load(file2)

    # Create a mapping of alternative_bom_id to product_pn and op_sequence
    bom_mapping = {}
    for entry in config2["prod_map_alt_assngs"]:
        alt_bom_id = entry["alternative_bom_id"]
        bom_mapping[alt_bom_id] = {"product_pn": entry["product_pn"], "op_sequence": entry["op_sequence"]}
        #print(bom_mapping)
    # Assign values to the required_asset_pn entries based on alternative_bom_id
    for entry in config1["prod_map_alt_boms"]:
        alt_bom_id = entry["alternative_bom_id"]
        if alt_bom_id in bom_mapping:
            entry["product_pn"] = bom_mapping[alt_bom_id]["product_pn"]
            entry["op_sequence"] = bom_mapping[alt_bom_id]["op_sequence"]

    # Save the modified config1 to a new file or overwrite the original file
    with open("compat_new_file.json", "w") as output_file:
        json.dump(config1, output_file, indent=2)


def assign_unique_configuration(json_path, extra_path):
    # Read JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        resource_ids = [i["required_asset_pn"] for i in data["prod_map_alt_boms"]]

        # Create a mapping of resource_id to unique integer

        # print(idx)
        unique_configurations = {resource_id: idx for idx, resource_id in enumerate(set(resource_ids))}

    # Assign unique integers to "required_asset_pn" and add "configuration" field
    for entry in data["prod_map_alt_boms"]:
        resource_id = entry['required_asset_pn']
        if resource_id in unique_configurations:
            unique_config = unique_configurations[resource_id]
            entry['required_asset_pn'] = f'T{unique_config}'
            entry['configuration'] = f'K{unique_config}'

    # Save the modified JSON data to a new file or overwrite the original file
    with open("compatible_new_file.json", "w") as output_file:
        json.dump(data, output_file, indent=2)

    ## Extracting the max from the unique configurations mapping
    max_value = max(unique_configurations.values())

    with open(extra_path, 'r') as json_file:
        extra_data = json.load(json_file)
        resource_final_ids = [i["mapping"] for i in extra_data["resource_code_mapping"]]

        # Create a mapping of resource_id to unique integer

        #print(resource_final_id)
        #print(idx + max_value)

        unique_final_configurations = {resource_final_id: idx + max_value for idx, resource_final_id in
                                       enumerate(set(resource_final_ids))}

    # Assign unique integers to "required_asset_pn" and add "configuration" field
    for entry in extra_data["resource_code_mapping"]:
        resource_id = entry['mapping']
        if resource_id in unique_final_configurations:
            unique_config = unique_final_configurations[resource_id]
            entry['mapping'] = f'T{unique_config}'
            entry['configuration'] = f'K{unique_config}'

    with open("extra_new_file.json", "w") as output_extra_file:
        json.dump(extra_data, output_extra_file, indent=2)



def divide_into_windows(file_path, output_dir):
    """
    Divides job data into 1-hour windows, assigns unique codes, and writes the output as JSON.

    Args:
        file_path (str): Path to the JSON file containing job data.
        output_file_path (str): Path to save the divided windows data in JSON format.
    """
    if not os.path.exists(output_dir):
    # If it doesn't exist, create it
       os.makedirs(output_dir)
       print(f"Folder '{output_dir}' was created.")
    else:
    # If it exists, print that it already exists
       print(f"Folder '{output_dir}' already exists.")
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        jobs = data["zeno_production_schedule"]
        windows = []
        current_window = []
        current_window_start_time = None

        for job in jobs:
            # Parse job_creation_date correctly
            date_str = job["job_imported_date"]
            job_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

            if current_window_start_time is None:
                current_window_start_time = job_time

            if job_time >= current_window_start_time + timedelta(hours = 1):
                windows.append({

                    "zeno_production_schedule": current_window
                })
                current_window = []
                current_window_start_time = job_time

            current_window.append(job)

        # Add the last window if it has any jobs
        if current_window:
            windows.append({

                "zeno_production_schedule": current_window
            })
        print(len(windows))
        for index,config in enumerate(windows):

            file_path = os.path.join(output_dir, f"D_{index}.json")
            with open(file_path, "w") as f:
                 json.dump(config, f, indent=4)



    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing file: {e}")






if __name__ == 'main':
    assign_values("prod_map_alt_boms_202408011658.json", "prod_map_alt_assngs_202408011658.json")
    assign_unique_configuration("compat_new_file.json", "resource_code_mapping_202408011658.json")
    divide_into_windows("production_plan_testnrf_filtered.json", "intermediate_files")
