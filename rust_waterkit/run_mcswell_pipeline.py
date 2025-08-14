# Run MCSwell -> receptor, project_path, box_center
# Create trajectory -> receptor, traj_path (project_path + frames/), out
# Run GIST -> path to gist_optimized.inp
# Run Hydration Sites identification -> path_to_grids

import os
import argparse
import subprocess

from run_waterkit import run_mcswell
from scripts.wk_make_trajectory import make_trajectory
from gridData import Grid
from waterkit.analysis import HydrationSites
from waterkit.analysis import blur_map

def create_gist_input(input_topology, input_traj, center):
    cmd = f"parm {input_topology}\n\
           trajin {input_traj}\n\
           gist gridspacn 0.5 gridcntr {center[0]} {center[1]} {center[2]} griddim 48 48 48\n\
           go\n\
           quit"
    with open("gist.inp", "w") as fo:
        fo.write(cmd)
    return

def run_hydration_sites_analysis(path_to_grids, out_path):
    path_to_go = os.path.join(path_to_grids, "gist-gO.dx")
    path_to_esw = os.path.join(path_to_grids, "gist-Esw-dens.dx")
    path_to_eww = os.path.join(path_to_grids, "gist-Eww-dens.dx")
    path_to_dtstrans = os.path.join(path_to_grids, "gist-dTStrans-dens.dx")
    path_to_dtsorient = os.path.join(path_to_grids, "gist-dTSorient-dens.dx")
    gO = Grid(path_to_go)
    esw = Grid(path_to_esw)
    eww = Grid(path_to_eww)
    tst = Grid(path_to_dtstrans)
    tso = Grid(path_to_dtsorient)
    dg = (esw + 2 * eww) - (tst + tso)

    # Identification of hydration site positions using gO
    hs = HydrationSites(gridsize=0.5, water_radius=1.4, min_water_distance=2.5, min_density=2.0)
    hydration_sites = hs.find(gO) # can pass "gist-gO.dx" directly also

    # Get Gaussian smoothed energy for hydration sites only
    dg_energy = hs.hydration_sites_energy(dg, water_radius=1.4)
    hs.export_to_pdb(os.path.join(out_path, "hydration_sites_dG_smoothed.pdb"), hydration_sites, dg_energy)

    # ... or get the whole Gaussian smoothed map
    map_smooth = blur_map(dg, radius=1.4)
    map_smooth.export(os.path.join(out_path, "gist-dG-dens_smoothed.dx"))
    return

def cmd_lineparser():
    parser = argparse.ArgumentParser(description='MCSwell pipeline')
    parser.add_argument('-r', dest='receptor_path', required=True,
                        action='store', help="Path to the receptor's file")
    parser.add_argument('-p', dest='project_path', required=True,
                        action='store', help='Path to where to save the results')
    parser.add_argument('-c', '--center', dest='box_center', nargs=3, type=float,
                        action='store', help='center of the box')
    # parser.add_argument("-a", dest='alg_type', required=True,
    #                     action='store', help='Algorithm to use. Choose between 1) gcmc 2) gcmcmc 3) gcmcsa')
    parser.add_argument('-traj_o', dest='traj_out', required=True,
                        action='store', help='Prefix of how to call the trajecotry files')
    # parser.add_argument('-g', dest='girds_path', required=True,
    #                     action='store', help='Path to grid files from GIST')
    parser.add_argument('-hso', dest='hydration_sites_out', required=True,
                        action='store', help='Out path for the hydration sites analysis')
    return parser.parse_args()

if __name__ == "__main__":
    args = cmd_lineparser()
    receptor_path = args.receptor_path
    base_project_path = args.project_path
    center = args.box_center
    # alg_type = args.alg_type
    trajectory_out_prefix = args.traj_out
    # grids_path = args.grids_path
    hydration_sites_out = args.hydration_sites_out

    # for alg_type in ["gcmc", "gcmcmc", "gcmcsa"]:
    for alg_type in ["gcmc"]:
        project_path = os.path.join(base_project_path, alg_type.upper())
        os.makedirs(project_path, exist_ok=True)
        # Run MCSwell first
        run_mcswell(receptor_path=receptor_path,
                    project_path=project_path,
                    center=center,
                    alg_type=alg_type)
        
        # os.chdir(f"{project_path}")
        # # Create trajectory from the frames
        # make_trajectory(receptor_filename=receptor_path,
        #                 water_directory=os.path.join(project_path, "frames"),
        #                 output_prefix=trajectory_out_prefix)
        
        # # Create GIST file and run it
        # input_topology = trajectory_out_prefix + "_system.prmtop"
        # input_traj = trajectory_out_prefix + "_system.nc"
        # create_gist_input(input_topology=input_topology,
        #                 input_traj=input_traj,
        #                 center=center)
        
        # gist_command = "cpptraj gist.inp"
        # process = subprocess.run(gist_command, shell=True, capture_output=True, text=True)
        # if process.returncode == 0:
        #     print("Command executed successfully:")
        #     print(process.stdout)
        # else:
        #     print("Command failed:")
        #     print(process.stderr)

        # # Hydration Sites Analysis
        # run_hydration_sites_analysis(path_to_grids=project_path,
        #                             out_path=os.path.join(hydration_sites_out, alg_type.upper()))