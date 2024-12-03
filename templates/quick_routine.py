import subprocess

def quick_routine_gpell(input_file, output_file):
    """
    Runs a bash command that takes an input file and writes to an output file.
    
    Parameters:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output text file.
    
    Returns:
        int: The exit code of the bash command. 0 indicates success.
    """
    # Construct the bash command
    # command = f"your_bash_command < {input_file} > {output_file}"
    command = f"/home/lic0a/src/geopsypack-src-3.3.3/build/bin/gpell < {input_file} > {output_file}"

    # Run the command
    #result = subprocess.run(command, shell=True, text=True)
    
    #command = [
    ##"gpell",
    #"/home/lic0a/src/geopsypack-src-3.3.3/build/bin/gpell",
    #"-fmin", "0.1",  # Minimum frequency
    #"-fmax", "30.0",  # Maximum frequency
    #"-step", "0.1",  # Frequency step
    #"-o", output_file,  # Output file
    #input_file  # Input model file
    #]

# Run the gpell command
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"Ellipticity curve successfully calculated and saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running gpell: {e}")
    
    return result.returncode
