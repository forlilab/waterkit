import rust_waterkit

def to_xyz(traj, step_size):
    with open(f"traj_{step_size}.xyz", 'w') as fo:
        fo.write(f"{len(traj)}\n")
        fo.write("\n")
        for t in traj:
            t_v = t
            fo.write(f"He {t_v[0]} {t_v[1]} {t_v[2]}\n")
    return 

if __name__ == "__main__":
    # select as, i. 111+107+103+162+150+98+97+184+96+93+55+52+51+138+139+136+135
    # lysine_atoms = [
    # rust_waterkit.Atom(atom_type='N', 
    #                    atom_id='', 
    #                    coords=[0.0, 0.0, 0.0], 
    #                    sigma=3.3, 
    #                    epsilon=0.2, 
    #                    charge=-0.3),  # Amine nitrogen
    # rust_waterkit.Atom(atom_type='C', 
    #                    atom_id='', 
    #                    coords=[1.5, 0.0, 0.0], 
    #                    sigma=3.5,
    #                    epsilon=0.1,
    #                    charge=0.1),  # Alpha carbon
    # rust_waterkit.Atom(atom_type='C', 
    #                    atom_id='', 
    #                    coords=[1.5, 1.5, 0.0], 
    #                    sigma=3.5,
    #                    epsilon=0.1,
    #                    charge=0.1),   # Side chain carbon
    # ]

    # water_atoms = [
    #     rust_waterkit.Atom(atom_type='O', 
    #                    atom_id='', 
    #                    coords=[3.0, 3.0, 3.0], 
    #                    sigma=3.0,
    #                    epsilon=0.3,
    #                    charge=-0.8),  # Oxygen
    #     rust_waterkit.Atom(atom_type='H', 
    #                    atom_id='', 
    #                    coords=[3.5, 3.0, 3.0], 
    #                    sigma=2.5,
    #                    epsilon=0.05,
    #                    charge=0.4),  # Hydrogen 1
    #     rust_waterkit.Atom(atom_type='H', 
    #                    atom_id='', 
    #                    coords=[2.5, 3.0, 3.0], 
    #                    sigma=2.5,
    #                    epsilon=0.05,
    #                    charge=0.4)  # Hydrogen 2
    # ]
    import time

    water_sphere = rust_waterkit.Sphere([-1.0, -1.0, 2.0])
    with open("/data/phd/waterkit/example/pocket.txt") as fi:
        lines = fi.readlines()
    
    surface_points = []
    for line in lines:
        line = line.strip().split(",")
        point = [float(line[3]), float(line[4]), float(line[5])]
        surface_points.append(point)

    start = time.time()
    step_size = 1.4
    trajectories = rust_waterkit.roll_sphere(surface_points, step_size)
    print(f"Time to grid: {time.time() - start}")
    to_xyz(trajectories, step_size)
    # energy = rust_waterkit.energy(lysine_atoms, water_atoms)
    # print(f"Energy computed with rust: {energy}")