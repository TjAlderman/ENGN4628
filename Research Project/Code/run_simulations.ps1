# Path to your Python script
$PYTHON_SCRIPT = "Research Project\Code\src\new_model.py"

# Function to run the simulation with given parameters
function Run-Simulation {
    param (
        $slope,
        $velocity,
        $acceleration,
        $time,
        $outputDir
    )

    Write-Host "Running simulation with parameters:"
    Write-Host "Slope: $slope, Velocity: $velocity, Acceleration: $acceleration, Time: $time"
    
    # Run the Python script with parameters
    python $PYTHON_SCRIPT $slope $velocity $acceleration $time
    
    # Move the output files to a specific directory
    Move-Item -Path "output.png" -Destination "$outputDir\output_${slope}_${velocity}_${acceleration}_${time}.png" -Force
    
    Write-Host "Simulation complete. Output saved in $outputDir"
    Write-Host "----------------------------------------"
}

# Create output directory
$OUTPUT_DIR = "simulation_results"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR

# Run simulations with different parameters
Run-Simulation "fake-slope.npy" "fake-velocity.npy" "fake-acceleration.npy" "fake-time.npy" $OUTPUT_DIR
# Run-Simulation "fake-slope2.npy" "fake-velocity2.npy" "fake-acceleration2.npy" "fake-time2.npy" $OUTPUT_DIR
# Run-Simulation "fake-slope3.npy" "fake-velocity3.npy" "fake-acceleration3.npy" "fake-time3.npy" $OUTPUT_DIR

Write-Host "All simulations completed."