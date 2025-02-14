import os
import subprocess
import sys
import shutil

if __name__ == "__main__":
    # base_path = "/home/niccolo/phd/waterkit/rust_waterkit/"
    base_path = "/data/phd/waterkit/rust_waterkit/"
    analysis_t = sys.argv[1]
    
    
    if analysis_t == "--optimized":
        mv_command = f"mv {os.path.join(base_path, 'test/*_optimized.pdb')} {os.path.join(base_path, 'test/optimized/traj/')}"
        process = subprocess.run(mv_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)
            
        # cd_cmd = f"cd {os.path.join(base_path, 'test/optimized/')}"
        os.chdir(f"{os.path.join(base_path, 'test/optimized/')}")
        
        make_traj_command = f"wk_make_trajectory.py -r {os.path.join(base_path, 'waterkit_data/1uyg_prepared.pdb')} -w traj/ -o 1uyg_optimized"
        process = subprocess.run(make_traj_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)
        gist_command = "cpptraj gist_optimized.inp"
        process = subprocess.run(gist_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)
            
    elif analysis_t == "--minimized":
        # mv_command = f"mv {os.path.join(base_path, 'test/*_unoptimized.pdb')} {os.path.join(base_path, 'test/unoptimized/traj/')}"
        # process = subprocess.run(mv_command, shell=True, capture_output=True, text=True)
        # if process.returncode == 0:
        #     print("Command executed successfully:")
        #     print(process.stdout)
        # else:
        #     print("Command failed:")
        #     print(process.stderr)
        # # subprocess.run(f"cd {os.path.join(base_path, 'test/unoptimized/')}", shell=True, capture_output=True, text=True)
        os.chdir(f"{os.path.join(base_path, 'test/unoptimized/')}")
        
        # make_traj_command = f"wk_make_trajectory.py -r {os.path.join(base_path, 'waterkit_data/1uyg_prepared.pdb')} -w traj/ -o 1uyg_unoptimized"
        # process = subprocess.run(make_traj_command, shell=True, capture_output=True, text=True)
        # if process.returncode == 0:
        #     print("Command executed successfully:")
        #     print(process.stdout)
        # else:
        #     print("Command failed:")
        #     print(process.stderr)
            
        gist_command = "cpptraj gist_unoptimized_minimized.inp"
        process = subprocess.run(gist_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)
            
    else:
        mv_command = f"mv {os.path.join(base_path, 'test/*_unoptimized.pdb')} {os.path.join(base_path, 'test/unoptimized/traj/')}"
        process = subprocess.run(mv_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)
        # subprocess.run(f"cd {os.path.join(base_path, 'test/unoptimized/')}", shell=True, capture_output=True, text=True)
        os.chdir(f"{os.path.join(base_path, 'test/unoptimized/')}")
        
        make_traj_command = f"wk_make_trajectory.py -r {os.path.join(base_path, 'waterkit_data/1uyg_prepared.pdb')} -w traj/ -o 1uyg_unoptimized"
        process = subprocess.run(make_traj_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)
            
        gist_command = "cpptraj gist_unoptimized.inp"
        process = subprocess.run(gist_command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Command executed successfully:")
            print(process.stdout)
        else:
            print("Command failed:")
            print(process.stderr)