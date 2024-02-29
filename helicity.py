import os 
import re

def run_helicity_predictions(sequences, path, name):
    
    # Create the directory structure
    if not os.path.exists(f'{path}/{name}'):
        os.makedirs(f'{path}/{name}')
        os.makedirs(f'{path}/{name}/seq/')
        os.makedirs(f'{path}/{name}/out/')

    # Prepare the input files for the predictions
    for idx, seq in enumerate(sequences):

        with open(f'{path}/{name}/seq/Seq_{idx}.seq', 'w') as f:
            f.write(f'>Seq_{idx}')
            f.write('\n')
            f.write(f'{seq}')

        with open(f'{path}/{name}_list', 'a') as f:
            f.write(f'Seq_{idx} ./{name}/seq/Seq_{idx}.seq')
            f.write('\n')
    
    # Update the script to run the predictions
    with open(f'{path}/run_spider3.sh', 'r') as file:
        script_contents = file.read()

        script_contents = re.sub(r"(SAVE_DIR=')(.*?)(?=')", rf"\1./{name}/out/", script_contents)
        script_contents = re.sub(r"(INPUT_LIST=')(.*?)(?=')", rf"\1./{name}_list", script_contents)

    with open(f'{path}/run_spider3.sh', 'w') as file:
        file.write(script_contents)
    
    # Run the bash script (if it was it was not run before)
    if len(os.listdir(f'{path}/{name}/out/')) == 0:
        os.system('bash spider3/run_spider3.sh')

    # Read the predictions
    predictions = []
    for idx, seq in enumerate(sequences):
        ss = []
        with open(f'{path}/{name}/out/Seq_{idx}.i1') as f:
            for line in f:
                line = line.strip()
                line = line.split(" ")
                ss.append(line[2])
        ss = ss[1:]
        predictions.append(ss)
    
    helicity = []
    for pred in predictions:
        if len(pred)!=0:
            helicity.append(pred.count('H')/len(pred))
        else:
            helicity.append(0)
    
    # Return the predictions
    return helicity