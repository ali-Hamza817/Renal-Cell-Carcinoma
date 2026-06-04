import os
import sys
import subprocess
import torch

def run_step(step_name, command_args):
    """Runs a pipeline step command and handles errors."""
    print(f"\n" + "="*80)
    print(f">>> PIPELINE STEP: {step_name.upper()}")
    print("="*80)
    
    python_exe = sys.executable
    cmd = [python_exe, "-u"] + command_args
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run process and stream stdout/stderr in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] Step '{step_name}' failed with exit code {process.returncode}.")
            sys.exit(process.returncode)
        else:
            print(f"\n[SUCCESS] Step '{step_name}' completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Error executing step '{step_name}': {e}")
        sys.exit(1)

def main():
    print("="*80)
    print("      ccRCC CT RISK STRATIFICATION PIPELINE ORCHESTRATOR      ")
    print("="*80)
    
    # 0. Hardware / Framework verification
    print("\n[INFO] Verification of Hardware and Deep Learning Framework:")
    print(f"  Python Version: {sys.version}")
    print(f"  PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("  [WARNING]: CUDA is not available. System will run on CPU, which is significantly slower.")
    
    # Define steps
    # Step 1: Download data (GDC Clinical Data & TCIA Image Subset)
    run_step("Data Download", ["src/data_download.py"])
    
    # Step 2: Preprocess scans (DICOM to normalized 3D volumes)
    run_step("Data Preprocessing", ["src/preprocess.py"])
    
    # Step 3: Visualize slices for validation
    run_step("Slice Visualization", ["visualize_slices.py"])
    
    # Step 4: Model Training
    model_choice = "resnet18"
    epochs = 15
    if len(sys.argv) > 1:
        model_choice = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            epochs = int(sys.argv[2])
        except ValueError:
            pass
            
    run_step("Model Training", ["src/train.py", "--model", model_choice, "--epochs", str(epochs)])
    
    # Step 5: Model Evaluation and Plotting
    run_step("Model Evaluation", ["src/evaluate.py"])
    
    print("\n" + "="*80)
    print("*** PIPELINE RUN COMPLETION: End-to-end model baseline is ready!")
    print("="*80)

if __name__ == "__main__":
    main()
